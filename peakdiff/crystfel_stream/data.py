#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import h5py
import signal
import msgpack
import numpy as np
import hashlib
import regex

import ray

from dataclasses import dataclass, field
from typing import Optional, Dict

from scipy.spatial.distance import cdist
from scipy.optimize         import linear_sum_assignment

from crystfel_stream_parser.engine import StreamParser
from crystfel_stream_parser.utils  import split_list_into_chunk

@dataclass
class StreamConfig:
    path_stream   : str
    path_cxi_root : str
    dir_output    : str
    num_cpus      : int
    cxi_key_data  : Optional[str] = '/entry_1/data_1/data'
    cxi_key_event : Optional[str] = '/LCLS/eventNumber'

class StreamManager:
    def __init__(self, config):
        self.path_stream   = config.path_stream
        self.path_cxi_root = config.path_cxi_root
        self.dir_output    = config.dir_output
        self.num_cpus      = config.num_cpus

        self.cxi_key  = {
            "data"  : config.cxi_key_data,
            "event" : config.cxi_key_event,
        }

        self.stream_data = self.parse_stream(self.num_cpus)

        self.misc = {
            "psana_exprun" : regex.compile(
                                 r"""(?x)
                                     (?&EXP)_r(?&RUN)

                                     (?(DEFINE)
                                         (?<EXP>
                                             [^/]+
                                         )
                                         (?<RUN>
                                             \d{4}
                                         )
                                     )
                                 """
                             )
        }


    def save_stream_as_msgpack(self, path_stream_msgpack, stream_data):
        print(f"Saving stream file from {path_stream_msgpack}...")
        stream_data_packed = msgpack.packb(stream_data)
        with open(path_stream_msgpack, 'wb') as f:
            f.write(stream_data_packed)


    def load_stream_from_msgpack(self, path_stream_msgpack):
        print(f"Loading stream file from {path_stream_msgpack}...")
        with open(path_stream_msgpack, 'rb') as f:
            data_packed = f.read()
            data = msgpack.unpackb(data_packed, strict_map_key = False)

        return data


    def parse_stream(self, num_cpus = 2):
        # Assign the new path to metrics...
        unique_identifier   = self.path_stream
        hashed_identifier   = hashlib.sha256(unique_identifier.encode()).hexdigest()
        path_stream_msgpack = os.path.join(self.dir_output, f"cache_{hashed_identifier}.stream.msgpack")

        # Check if cache is available???
        if not os.path.exists(path_stream_msgpack):
            # Parse from scratch...
            stream_data = StreamParser(self.path_stream).parse(num_cpus = num_cpus)

            # Save them...
            self.save_stream_as_msgpack(path_stream_msgpack, stream_data)

        else:
            # Load it from cache...
            stream_data = self.load_stream_from_msgpack(path_stream_msgpack)

        return stream_data


    def get_num_peaks(self):
        '''
        Returns a generator.
        '''
        return ( int(self.stream_data[i]['metadata']['num_peaks']) for i, _ in enumerate(self.stream_data) )


    def get_indexed_frame(self):
        '''
        Returns a generator.
        '''
        return ( i for i, _ in enumerate(self.stream_data) if len(self.stream_data[i]['crystal']) > 0 )


    def get_found_peaks(self, seqi):
        '''
        fs -> x
        ss -> y
        '''
        peaks = [ peak for peaks in self.stream_data[seqi]['found peaks'].values() for peak in peaks ]
        peaks = [ (y, x) for x, y, _, _, in peaks ]

        return peaks


    def get_predicted_peaks(self, seqi, sigma_cut = float('-inf')):
        peaks = [ peak for crystal in self.stream_data[seqi]['crystal']
                           for peaks in crystal['predicted peaks'].values()
                               for peak in peaks ]
        peaks = [ (y, x) for h, k, l, intensity, sigma, max_peak, background, x, y in peaks if intensity/sigma >= sigma_cut]

        return peaks


    def split_peaks_by_sigma(self, seqi, sigma_cut = float('-inf')):
        peaks = [ peak for crystal in self.stream_data[seqi]['crystal']
                           for peaks in crystal['predicted peaks'].values()
                               for peak in peaks ]
        good_peaks = []
        bad_peaks  = []
        for h, k, l, intensity, sigma, max_peak, background, x, y in peaks:
            if intensity/sigma >= sigma_cut:
                good_peaks.append((y, x))
            else:
                bad_peaks.append((y, x))

        return good_peaks, bad_peaks


    def get_img(self, seqi):
        path_cxi_root = self.path_cxi_root
        path_cxi      = self.stream_data[seqi]['metadata']['Image filename']
        path_cxi      = os.path.join(path_cxi_root, path_cxi)

        idx_in_cxi    = self.stream_data[seqi]['metadata']['Event'][2:]    # '//375'
        idx_in_cxi    = int(idx_in_cxi)

        key_img   = self.cxi_key["data"]
        with h5py.File(path_cxi, 'r') as fh:
            img = fh.get(key_img)[idx_in_cxi][()]

        return img


    def get_psana_event_tuple(self, seqi):
        path_cxi_root      = self.path_cxi_root
        path_cxi_in_stream = self.stream_data[seqi]['metadata']['Image filename']
        path_cxi           = os.path.join(path_cxi_root, path_cxi_in_stream)

        # Try to figure out the (exp, run) otherwise just use None...
        exp, run = None, None
        psana_exprun_pattern = self.misc["psana_exprun"]
        match = regex.search(psana_exprun_pattern, path_cxi_in_stream)
        if match is not None:
            capture_dict = match.capturesdict()
            exp = capture_dict["EXP"][0]
            run = capture_dict["RUN"][0]

        idx_in_cxi = self.stream_data[seqi]['metadata']['Event'][2:]    # '//375'
        idx_in_cxi = int(idx_in_cxi)

        key_psana_event_idx = self.cxi_key["event"]
        with h5py.File(path_cxi, 'r') as fh:
            psana_event_idx = fh.get(key_psana_event_idx)[idx_in_cxi][()]

        return (exp, run, psana_event_idx)




@dataclass
class StreamPeakDiffConfig:
    stream_config : StreamConfig
    dir_output    : Optional[str] = 'peakdiff_results'

class StreamPeakDiff:
    """
    Perform peakdiff on a crystfel stream file.
    """
    def __init__(self, config):
        super().__init__()

        self.stream_config = config.stream_config
        self.dir_output    = config.dir_output

        self.stream_manager = StreamManager(self.stream_config)

        os.makedirs(self.dir_output, exist_ok=True)


    @staticmethod
    def peakdiff(found_peaks, predicted_peaks, threshold_distance = 5):
        # Compute a cost matrix...
        found_peaks     = np.array(found_peaks)
        predicted_peaks = np.array(predicted_peaks)
        cost_matrix = cdist(found_peaks, predicted_peaks)

        # Use linear_sum_assignment to find the optimal assignment...
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter assignments based on some threshold distance...
        common_peaks = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] <= threshold_distance]

        return common_peaks


    def compute_metrics(self, num_cpus, threshold_distance = 5, uses_ray_put = False):
        """
        Currently support single node.
        """
        stream_manager = self.stream_manager

        # Shutdown ray clients during a Ctrl+C event...
        def signal_handler(sig, frame):
            if ray.is_initialized():
                print('SIGINT (Ctrl+C) caught, shutting down Ray...')
                ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Init ray...
        ray.init(num_cpus = num_cpus)

        # Build a list of input frame and metadata to process...
        # ...Fetch indexed frames
        indexed_frame_idx_list = stream_manager.get_indexed_frame()

        # ...Build a list of input event and metadata
        event_data = [ {
            'frame_idx'       : frame_idx,
            'metadata'        : stream_manager.stream_data[frame_idx]['metadata'],
            'found peaks'     : stream_manager.get_found_peaks(frame_idx),
            'predicted peaks' : stream_manager.get_predicted_peaks(frame_idx),
        } for frame_idx in indexed_frame_idx_list ]

        # Create the procedure of processing one event...
        def process_event(event_data, threshold_distance = 5):
            # Unpack input...
            frame_idx       = event_data['frame_idx'      ]
            metadata        = event_data['metadata'       ]
            found_peaks     = event_data['found peaks'    ]    # [(y, x), ...]
            predicted_peaks = event_data['predicted peaks']    # [(y, x), ...]

            common_peaks = StreamPeakDiff.peakdiff(found_peaks, predicted_peaks, threshold_distance)

            ## # Compute a cost matrix...
            ## found_peaks     = np.array(found_peaks)
            ## predicted_peaks = np.array(predicted_peaks)
            ## cost_matrix = cdist(found_peaks, predicted_peaks)

            ## # Use linear_sum_assignment to find the optimal assignment...
            ## row_ind, col_ind = linear_sum_assignment(cost_matrix)

            ## # Filter assignments based on some threshold distance...
            ## common_peaks = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] <= threshold_distance]

            # Calculate metrics...
            # Assume 'common peaks' as 'true peaks'
            num_tp = len(common_peaks)              # ...True positive
            num_fp = len(found_peaks) - num_tp     # ...False positive
            num_fn = len(predicted_peaks) - num_tp # ...False negative
            precision = num_tp / (num_tp + num_fp) # ...Coverage of common peaks in found peaks
            recall    = num_tp / (num_tp + num_fn) # ...Coverage of common peaks in predicted_peaks

            return {
                ## "metadata"  : (frame_idx, metadata['Image filename'], metadata['Event']),
                "frame_idx"       : int(frame_idx),
                "found_peaks"     : int(len(found_peaks)),
                "predicted_peaks" : int(len(predicted_peaks)),
                ## "common_peaks"    : int(len(common_peaks)),
                "num_tp"          : int(num_tp),
                "num_fp"          : int(num_fp),
                "num_fn"          : int(num_fn),
                "precision"       : float(precision),
                "recall"          : float(recall),
            }

        # Create the procedure of processing a list of events...
        @ray.remote
        def process_batch_of_events(event_data_list, threshold_distance):
            results = []
            for event_data in event_data_list:
                result = process_event(event_data, threshold_distance)
                results.append(result)
            return results

        # Chunking the input by grouping events...
        batch_size = num_cpus
        batches = split_list_into_chunk(event_data, max_num_chunk = num_cpus)

        # Optional ray put for saving memory???
        if uses_ray_put:
            batches = [ray.put(batch) for batch in batches]

        # Submit the computation jobs at remote nodes and they will start now...
        futures = [process_batch_of_events.remote(batch, threshold_distance) for batch in batches]

        peakdiff_result = dict(
            num_tp          = {},
            num_fp          = {},
            num_fn          = {},
            found_peaks     = {},
            predicted_peaks = {},
            precision       = {},
            recall          = {},
        )
        remaining_futures = futures
        while remaining_futures:
            # Wait for at least one task to be ready
            ready_futures, remaining_futures = ray.wait(remaining_futures, num_returns=1, timeout=None)

            # Fetch the result of the ready task(s)
            for future in ready_futures:
                result_per_worker = ray.get(future)
                for result_dict in result_per_worker:
                    frame_idx = result_dict["frame_idx"]

                    for k in peakdiff_result.keys():
                        peakdiff_result[k][frame_idx] = result_dict[k]

        ray.shutdown()

        return peakdiff_result


    def save_metrics_as_msgpack(self, path_metrics, metrics):
        print(f"Saving metrics file from {path_metrics}...")
        metrics_packed = msgpack.packb(metrics)
        with open(path_metrics, 'wb') as f:
            f.write(metrics_packed)


    def load_metrics_from_msgpack(self, path_metrics):
        print(f"Loading metrics file from {path_metrics}...")
        with open(path_metrics, 'rb') as f:
            data_packed = f.read()
            data = msgpack.unpackb(data_packed, strict_map_key = False)

        return data


    def build_bokeh_data_source(self, path_metrics = None, num_cpus = 20):
        # Generate a unique path???
        if path_metrics is None:
            # Assign the new path to metrics...
            unique_identifier = self.stream_config.path_stream
            hashed_identifier = hashlib.sha256(unique_identifier.encode()).hexdigest()
            path_metrics = os.path.join(self.dir_output, f"cache_{hashed_identifier}.peakdiff.msgpack")

        # Check if cache is available???
        if not os.path.exists(path_metrics):
            # Compute the metrics...
            metrics = self.compute_metrics(num_cpus = num_cpus)

            # Save them...
            self.save_metrics_as_msgpack(path_metrics, metrics)

        else:
            # Load it from cache...
            metrics = self.load_metrics_from_msgpack(path_metrics)

        # Build bokeh data source...
        num_tp          = metrics["num_tp"         ]
        num_fp          = metrics["num_fp"         ]
        num_fn          = metrics["num_fn"         ]
        found_peaks     = metrics["found_peaks"    ]
        predicted_peaks = metrics["predicted_peaks"]
        precision       = metrics["precision"      ]
        recall          = metrics["recall"         ]
        data_source = dict(
            events              = list(num_tp.keys()),    # ...Confusingly, just metadata
            num_found_peaks     = list(found_peaks.values()),
            num_predicted_peaks = list(predicted_peaks.values()),
            precision           = list(precision.values()),
            recall              = list(recall.values()),
        )

        return data_source
