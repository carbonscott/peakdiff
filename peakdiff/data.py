#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import h5py
import signal
import msgpack
import numpy as np
import hashlib

from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize         import linear_sum_assignment

import ray


class CXIPeakDiff:
    """
    Perform peakdiff on two cxi files.
    """
    def __init__(self, path_cxi_0, path_cxi_1, dir_output = None):
        super().__init__()

        self.path_cxi_0 = path_cxi_0
        self.path_cxi_1 = path_cxi_1
        self.dir_output = 'peakdiff_results' if dir_output is None else dir_output

        os.makedirs(self.dir_output, exist_ok=True)


    def get_n_peaks(self, path_cxi):
        with h5py.File(path_cxi, 'r') as fh:
            n_peaks = fh.get('entry_1/result_1/nPeaksAll')[:]

        return n_peaks


    def compute_metrics(self, num_cpus, min_n_peaks = 10, threshold_distance = 5, uses_ray_put = False):
        """
        Currently support single node.
        """
        # Shutdown ray clients during a Ctrl+C event...
        def signal_handler(sig, frame):
            print('SIGINT (Ctrl+C) caught, shutting down Ray...')
            ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Init ray...
        ray.init(num_cpus = num_cpus)

        # Build a list of input event and metadata to process...
        # ...Set path
        path_cxi_0 = self.path_cxi_0
        path_cxi_1 = self.path_cxi_1

        # ...Fetch num of peaks
        n_peaks_0 = self.get_n_peaks(path_cxi_0)
        n_peaks_1 = self.get_n_peaks(path_cxi_1)

        # ...Keep events having no less than min number of peaks
        n_peaks_0 = { enum_idx : n for enum_idx, n in enumerate(n_peaks_0) if n >= min_n_peaks }
        n_peaks_1 = { enum_idx : n for enum_idx, n in enumerate(n_peaks_1) if n >= min_n_peaks }

        # ...Find common events
        common_events = list(set([event for event in n_peaks_0.keys()]) & set([event for event in n_peaks_1.keys()]))

        # ...Build a list of input event and metadata
        event_data = []
        for event in common_events:
            event_data.append(
                {
                    "event"     : event,
                    "path_cxi_0": path_cxi_0,
                    "path_cxi_1": path_cxi_1,
                    "n_peaks_0" : n_peaks_0[event],
                    "n_peaks_1" : n_peaks_1[event],
                }
            )

        # Create the procedure of processing one event...
        def process_event(event_data, threshold_distance = 5):
            # Unpack input...
            event      = event_data["event"]
            path_cxi_0 = event_data["path_cxi_0"]
            path_cxi_1 = event_data["path_cxi_1"]
            n_peaks_0  = event_data["n_peaks_0"]
            n_peaks_1  = event_data["n_peaks_1"]

            # Fetch coordinates...
            with h5py.File(path_cxi_0, "r") as fh:
                peaks_y_0 = fh.get('entry_1/result_1/peakYPosRawAll')[event][:n_peaks_0]
                peaks_x_0 = fh.get('entry_1/result_1/peakXPosRawAll')[event][:n_peaks_0]

            with h5py.File(path_cxi_1, "r") as fh:
                peaks_y_1 = fh.get('entry_1/result_1/peakYPosRawAll')[event][:n_peaks_1]
                peaks_x_1 = fh.get('entry_1/result_1/peakXPosRawAll')[event][:n_peaks_1]

            # Format into [(y0, x0), (y1, x1), ..., (y_n, x_n)]...
            peaks_0 = [ (y, x) for y, x in zip(peaks_y_0, peaks_x_0) ]
            peaks_1 = [ (y, x) for y, x in zip(peaks_y_1, peaks_x_1) ]

            # Compute a cost matrix...
            coords_0    = np.array(peaks_0)
            coords_1    = np.array(peaks_1)
            cost_matrix = cdist(coords_0, coords_1)

            # Use linear_sum_assignment to find the optimal assignment...
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Filter assignments based on some threshold distance...
            close_pairs = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] <= threshold_distance]

            # Calculate match rate...
            match_rate_0 = len(close_pairs) / len(peaks_0)
            match_rate_1 = len(close_pairs) / len(peaks_1)

            return {
                "event"        : event,
                "match_rate_0" : match_rate_0,
                "match_rate_1" : match_rate_1,
                "n_peaks_0"    : len(peaks_0),
                "n_peaks_1"    : len(peaks_1),
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
        batches = [event_data[i:i + batch_size] for i in range(0, len(event_data), batch_size)]

        # Optional ray put for saving memory???
        if uses_ray_put:
            batches = [ray.put(batch) for batch in batches]

        # Register the computation at remote nodes...
        results = [process_batch_of_events.remote(batch, threshold_distance) for batch in batches]

        # Compute...
        results = ray.get(results)

        # Collect results...
        match_rate_0 = {}
        match_rate_1 = {}
        n_peaks_0    = {}
        n_peaks_1    = {}
        for batch_result in results:
            for result_dict in batch_result:
                event = result_dict["event"]

                match_rate_0[event] = result_dict["match_rate_0"]
                match_rate_1[event] = result_dict["match_rate_1"]
                n_peaks_0   [event] = result_dict["n_peaks_0"   ]
                n_peaks_1   [event] = result_dict["n_peaks_1"   ]

        # Shutdown ray...
        ray.shutdown()

        return {
            "match_rate_0" : match_rate_0,
            "match_rate_1" : match_rate_1,
            "n_peaks_0"    : n_peaks_0,
            "n_peaks_1"    : n_peaks_1,
        }


    def save_metrics_as_msgpack(self, path_metrics, metrics):
        metrics_packed = msgpack.packb(metrics)
        with open(path_metrics, 'wb') as f:
            f.write(metrics_packed)


    def load_metrics_from_msgpack(self, path_metrics):
        with open(path_metrics, 'rb') as f:
            data_packed = f.read()
            data = msgpack.unpackb(data_packed, strict_map_key = False)

        return data


    def build_bokeh_data_source(self, path_metrics = None, num_cpus = 20):
        # Generate a unique path???
        if path_metrics is None:
            # Set path...
            path_cxi_0 = self.path_cxi_0
            path_cxi_1 = self.path_cxi_1

            # Generate a unique id...
            unique_identifier = f"{path_cxi_0}.{path_cxi_1}"
            hashed_identifier = hashlib.sha256(unique_identifier.encode()).hexdigest()

            # Assign the new path to metrics...
            path_metrics = os.path.join(self.dir_output, f"cache_{hashed_identifier}.msgpack")

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
        file_cxi_0   = os.path.basename(path_cxi_0)
        file_cxi_1   = os.path.basename(path_cxi_1)
        match_rate_0 = metrics["match_rate_0"]
        match_rate_1 = metrics["match_rate_1"]
        n_peaks_0    = metrics["n_peaks_0"   ]
        n_peaks_1    = metrics["n_peaks_1"   ]
        data_source = dict(
            events = list(n_peaks_0.keys()),
            n_peaks_x = list(n_peaks_0.values()),
            n_peaks_y = list(n_peaks_1.values()),
            n_peaks_l = [f"event {event:06d}, {file_cxi_0}:{m_0:.2f}, {file_cxi_1}:{m_1:.2f}"
                         for event, m_0, m_1 in zip(n_peaks_0.keys(),
                                                    n_peaks_0.values(),
                                                    n_peaks_1.values())],
            match_rates_x = list(match_rate_0.values()),
            match_rates_y = list(match_rate_1.values()),
            match_rates_l = [f"event {event:06d}, {file_cxi_0}:{m_0:.2f}, {file_cxi_1}:{m_1:.2f}"
                            for event, m_0, m_1 in zip(match_rate_0.keys(),
                                                       match_rate_0.values(),
                                                       match_rate_1.values())],
        )

        return data_source
