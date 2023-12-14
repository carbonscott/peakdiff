#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import h5py
import signal
import msgpack
import numpy as np
import hashlib

from dataclasses import dataclass, field

from typing import Optional, Dict

from scipy.spatial.distance import cdist
from scipy.optimize         import linear_sum_assignment

import ray


@dataclass
class CXIKeyConfig:
    num_peaks  : Optional[str] = "/entry_1/result_1/nPeaks",
    peak_event : Optional[str] = "/entry_1/result_1/peakEvent",
    peak_y     : Optional[str] = "/entry_1/result_1/peakYPosRaw",
    peak_x     : Optional[str] = "/entry_1/result_1/peakXPosRaw",
    data       : Optional[str] = "/entry_1/data_1/data",
    mask       : Optional[str] = "/entry_1/data_1/mask",
    segmask    : Optional[str] = "/entry_1/data_1/segmask",


@dataclass
class CXIConfig:
    peakfinder : str    # peaknet, pyalgo, pf8
    path_cxi   : str
    key        : CXIKeyConfig


class CXIManager:
    def __init__(self, config):
        self.path_cxi   = config.path_cxi
        self.key        = config.key
        self.peakfinder = config.peakfinder

        self._event_to_seqi = None


    def get_n_peaks(self):
        path_cxi = self.path_cxi

        key_num_peaks = self.key.num_peaks
        with h5py.File(path_cxi, "r") as fh:
            n_peaks = fh.get(key_num_peaks)[:]

        return n_peaks


    def get_img_by_event(self, event):
        path_cxi = self.path_cxi
        key_data = self.key.data

        img = None
        if self._event_to_seqi is None:
            key_num_peaks = self.key.num_peaks
            with h5py.File(self.path_cxi, "r") as fh:
                if key_num_peaks in fh:
                    num_peaks = fh.get(self.key.num_peaks)[:]
                    self._event_to_seqi = { event : seqi for seqi, event in enumerate(np.where(num_peaks > -1)[0]) }

        seqi = self._event_to_seqi[event]
        with h5py.File(path_cxi, "r") as fh:
            if key_data in fh:
                img_dataset = fh.get(key_data)
                if seqi < len(img_dataset):
                    img = img_dataset[seqi]

        return img


    def get_peaks_by_event(self, event, max_num_peaks = None):
        path_cxi   = self.path_cxi

        key_peak_y = self.key.peak_y
        key_peak_x = self.key.peak_x

        if max_num_peaks is None:
            n_peaks          = self.get_n_peaks()
            n_peaks_by_event = n_peaks[event]
            max_num_peaks    = n_peaks_by_event

        with h5py.File(path_cxi, "r") as fh:
            peaks_y = fh.get(key_peak_y)[event][:max_num_peaks]
            peaks_x = fh.get(key_peak_x)[event][:max_num_peaks]

        return peaks_y, peaks_x


@dataclass
class CXIPeakDiffConfig:
    cxi_config_0   : CXIConfig
    cxi_config_1   : CXIConfig
    uses_cxi_0_img : bool
    dir_output     : Optional[str] = 'peakdiff_results'


class CXIPeakDiff:
    """
    Perform peakdiff on two cxi files.
    """
    def __init__(self, config):
        super().__init__()

        self.cxi_config_0   = config.cxi_config_0
        self.cxi_config_1   = config.cxi_config_1
        self.uses_cxi_0_img = config.uses_cxi_0_img
        self.dir_output     = config.dir_output

        self.cxi_manager_0 = CXIManager(self.cxi_config_0)
        self.cxi_manager_1 = CXIManager(self.cxi_config_1)


        os.makedirs(self.dir_output, exist_ok=True)


    def compute_metrics(self, num_cpus, min_n_peaks = 10, threshold_distance = 5, uses_ray_put = False):
        """
        Currently support single node.
        """
        cxi_config_0 = self.cxi_config_0
        cxi_config_1 = self.cxi_config_1

        # Shutdown ray clients during a Ctrl+C event...
        def signal_handler(sig, frame):
            print('SIGINT (Ctrl+C) caught, shutting down Ray...')
            ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Init ray...
        ray.init(num_cpus = num_cpus)

        # Build a list of input event and metadata to process...
        # ...Fetch num of peaks
        n_peaks_0 = self.cxi_manager_0.get_n_peaks()
        n_peaks_1 = self.cxi_manager_1.get_n_peaks()

        # ...Keep events having no less than min number of peaks
        n_peaks_0_filtered = { event : n for event, n in enumerate(n_peaks_0) if n >= min_n_peaks }
        n_peaks_1_filtered = { event : n for event, n in enumerate(n_peaks_1) if n >= min_n_peaks }

        # ...Find common events
        common_events = list(set([event for event in n_peaks_0_filtered.keys()]) & set([event for event in n_peaks_1_filtered.keys()]))

        # ...Build a list of input event and metadata
        event_data = []
        for event in common_events:
            # ...Get num of peaks by event
            n_peaks_by_event_0 = n_peaks_0_filtered[event]
            n_peaks_by_event_1 = n_peaks_1_filtered[event]

            event_data.append(
                {
                    "event"              : event,
                    "cxi_manager_0"      : self.cxi_manager_0,
                    "cxi_manager_1"      : self.cxi_manager_1,
                    "n_peaks_by_event_0" : n_peaks_by_event_0,
                    "n_peaks_by_event_1" : n_peaks_by_event_1,
                }
            )

        # Create the procedure of processing one event...
        def process_event(event_data, threshold_distance = 5):
            # Unpack input...
            event               = event_data["event"]
            cxi_manager_0       = event_data["cxi_manager_0"]
            cxi_manager_1       = event_data["cxi_manager_1"]
            n_peaks_by_event_0  = event_data["n_peaks_by_event_0"]
            n_peaks_by_event_1  = event_data["n_peaks_by_event_1"]

            # Fetch coordinates...
            peaks_y_0, peaks_x_0 = cxi_manager_0.get_peaks_by_event(event, max_num_peaks = n_peaks_by_event_0)
            peaks_y_1, peaks_x_1 = cxi_manager_1.get_peaks_by_event(event, max_num_peaks = n_peaks_by_event_1)

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
            m_rates_by_event_0 = len(close_pairs) / len(peaks_0)
            m_rates_by_event_1 = len(close_pairs) / len(peaks_1)

            return {
                "event"              : int(event),
                "m_rates_by_event_0" : float(m_rates_by_event_0),
                "m_rates_by_event_1" : float(m_rates_by_event_1),
                "n_peaks_by_event_0" : int(n_peaks_by_event_0),
                "n_peaks_by_event_1" : int(n_peaks_by_event_1),
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
        m_rates_0 = {}
        m_rates_1 = {}
        n_peaks_0 = {}
        n_peaks_1 = {}
        for batch_result in results:
            for result_dict in batch_result:
                event = result_dict["event"]

                m_rates_0[event] = result_dict["m_rates_by_event_0"]
                m_rates_1[event] = result_dict["m_rates_by_event_1"]
                n_peaks_0[event] = result_dict["n_peaks_by_event_0"]
                n_peaks_1[event] = result_dict["n_peaks_by_event_1"]

        # Shutdown ray...
        ray.shutdown()

        return {
            "m_rates_0" : m_rates_0,
            "m_rates_1" : m_rates_1,
            "n_peaks_0" : n_peaks_0,
            "n_peaks_1" : n_peaks_1,
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
            path_cxi_0 = self.cxi_config_0.path_cxi
            path_cxi_1 = self.cxi_config_1.path_cxi

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
        file_cxi_config_0 = os.path.basename(path_cxi_0)
        file_cxi_config_1 = os.path.basename(path_cxi_1)
        m_rates_0  = metrics["m_rates_0"]
        m_rates_1  = metrics["m_rates_1"]
        n_peaks_0  = metrics["n_peaks_0"]
        n_peaks_1  = metrics["n_peaks_1"]
        data_source = dict(
            events = list(n_peaks_0.keys()),
            n_peaks_x = list(n_peaks_0.values()),
            n_peaks_y = list(n_peaks_1.values()),
            n_peaks_l = [f"event {event:06d}, {file_cxi_config_0}:{m_0:.2f}, {file_cxi_config_1}:{m_1:.2f}"
                         for event, m_0, m_1 in zip(n_peaks_0.keys(),
                                                    n_peaks_0.values(),
                                                    n_peaks_1.values())],
            m_rates_x = list(m_rates_0.values()),
            m_rates_y = list(m_rates_1.values()),
            m_rates_l = [f"event {event:06d}, {file_cxi_config_0}:{m_0:.2f}, {file_cxi_config_1}:{m_1:.2f}"
                        for event, m_0, m_1 in zip(m_rates_0.keys(),
                                                   m_rates_0.values(),
                                                   m_rates_1.values())],
        )

        return data_source
