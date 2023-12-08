#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bokeh.models import ColumnDataSource

import os
import sys
import h5py
import signal
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import ray


class CXIPeakDiff:
    """
    Perform peakdiff on two cxi files.
    """
    def __init__(self, path_cxi_0, path_cxi_1):
        super().__init__()

        self.path_cxi_0 = path_cxi_0
        self.path_cxi_1 = path_cxi_1

        self.min_n_peaks = 10
        self.threshold_distance = 5


    def get_n_peaks(self, path_cxi):
        with h5py.File(path_cxi, 'r') as fh:
            n_peaks = fh.get('entry_1/result_1/nPeaksAll')[:]

        return n_peaks


    def compute_metrics(self, num_cpus, threshold_distance = 5):
        """
        Currently support single node.
        """
        def signal_handler(sig, frame):
            print('SIGINT (Ctrl+C) caught, shutting down Ray...')
            ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        ray.init(num_cpus = num_cpus)

        path_cxi_0 = self.path_cxi_0
        path_cxi_1 = self.path_cxi_1

        min_n_peaks = self.min_n_peaks

        n_peaks_0 = self.get_n_peaks(path_cxi_0)
        n_peaks_1 = self.get_n_peaks(path_cxi_1)

        n_peaks_0 = { enum_idx : n for enum_idx, n in enumerate(n_peaks_0) if n >= min_n_peaks }
        n_peaks_1 = { enum_idx : n for enum_idx, n in enumerate(n_peaks_1) if n >= min_n_peaks }

        common_events = list(set([event for event in n_peaks_0.keys()]) & set([event for event in n_peaks_1.keys()]))

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

        def process_event(event_data, threshold_distance = 5):
            event      = event_data["event"]
            path_cxi_0 = event_data["path_cxi_0"]
            path_cxi_1 = event_data["path_cxi_1"]
            n_peaks_0  = event_data["n_peaks_0"]
            n_peaks_1  = event_data["n_peaks_1"]

            with h5py.File(path_cxi_0, "r") as fh:
                peaks_y_0 = fh.get('entry_1/result_1/peakYPosRawAll')[event][:n_peaks_0]
                peaks_x_0 = fh.get('entry_1/result_1/peakXPosRawAll')[event][:n_peaks_0]

            with h5py.File(path_cxi_1, "r") as fh:
                peaks_y_1 = fh.get('entry_1/result_1/peakYPosRawAll')[event][:n_peaks_1]
                peaks_x_1 = fh.get('entry_1/result_1/peakXPosRawAll')[event][:n_peaks_1]

            peaks_0 = [ (y, x) for y, x in zip(peaks_y_0, peaks_x_0) ]
            peaks_1 = [ (y, x) for y, x in zip(peaks_y_1, peaks_x_1) ]

            # Assume coords_0 and coords_1 are your two lists of coordinates
            coords_0 = np.array(peaks_0)
            coords_1 = np.array(peaks_1)

            # Compute a cost matrix.
            cost_matrix = cdist(coords_0, coords_1)

            # Use linear_sum_assignment to find the optimal assignment.
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Filter assignments based on some threshold distance.
            close_pairs = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] <= threshold_distance]

            match_rate_0  = len(close_pairs) / len(peaks_0)
            match_rate_1 = len(close_pairs) / len(peaks_1)

            return {
                "event"        : event,
                "match_rate_0" : match_rate_0,
                "match_rate_1" : match_rate_1,
                "n_peaks_0"    : len(peaks_0),
                "n_peaks_1"    : len(peaks_1),
            }

        @ray.remote
        def process_batch_of_events(event_data_list, threshold_distance):
            results = []
            for event_data in event_data_list:
                result = process_event(event_data, threshold_distance)
                results.append(result)
            return results

        batch_size = num_cpus
        batches = [ray.put(event_data[i:i + batch_size]) for i in range(0, len(event_data), batch_size)]

        threshold_distance = self.threshold_distance
        results = [process_batch_of_events.remote(batch, threshold_distance) for batch in batches]

        match_rate_0 = {}
        match_rate_1 = {}
        n_peaks_0    = {}
        n_peaks_1    = {}

        results = ray.get(results)
        for batch_result in results:
            for result_dict in batch_result:
                event = result_dict["event"]

                match_rate_0[event] = result_dict["match_rate_0"]
                match_rate_1[event] = result_dict["match_rate_1"]
                n_peaks_0   [event] = result_dict["n_peaks_0"   ]
                n_peaks_1   [event] = result_dict["n_peaks_1"   ]

        ray.shutdown()

        return {
            "match_rate_0" : match_rate_0,
            "match_rate_1" : match_rate_1,
            "n_peaks_0"    : n_peaks_0,
            "n_peaks_1"    : n_peaks_1,
        }




class DataSource:

    def __init__(self, config = None):

        self.path_n_peaks_A = config.path_n_peaks_A
        self.path_m_rates_A = config.path_m_rates_A
        self.path_n_peaks_B = config.path_n_peaks_B
        self.path_m_rates_B = config.path_m_rates_B

        # Read the data to visualize...
        n_peaks_dict = {}
        m_rates_dict = {}
        with open(config.path_n_peaks_A, 'rb') as f:
            data = f.read()
            n_peaks_dict['peaknet'] = msgpack.unpackb(data, strict_map_key = False)

        with open(config.path_n_peaks_B, 'rb') as f:
            data = f.read()
            n_peaks_dict['pyalgo'] = msgpack.unpackb(data, strict_map_key = False)

        with open(config.path_m_rates_A, 'rb') as f:
            data = f.read()
            m_rates_dict['peaknet'] = msgpack.unpackb(data, strict_map_key = False)

        with open(config.path_m_rates_B, 'rb') as f:
            data = f.read()
            m_rates_dict['pyalgo'] = msgpack.unpackb(data, strict_map_key = False)

        # Build the data source (it's pandas dataframe under the hood)...
        data_source = dict(
            events    = list(n_peaks_dict['peaknet'].keys()),
            n_peaks_x = list(n_peaks_dict['pyalgo'].values()),
            n_peaks_y = list(n_peaks_dict['peaknet'].values()),
            n_peaks_l = [f"event {event:06d}, pyalgo:{m_pyalgo:.2f}, peaknet:{m_peaknet:.2f}"
                         for event, m_pyalgo, m_peaknet in zip(n_peaks_dict['pyalgo' ].keys(),
                                                               n_peaks_dict['pyalgo' ].values(),
                                                               n_peaks_dict['peaknet'].values())],
            m_rates_x = list(m_rates_dict['pyalgo'].values()),
            m_rates_y = list(m_rates_dict['peaknet'].values()),
            m_rates_l = [f"event {event:06d}, pyalgo:{m_pyalgo:.2f}, peaknet:{m_peaknet:.2f}"
                         for event, m_pyalgo, m_peaknet in zip(m_rates_dict['pyalgo' ].keys(),
                                                               m_rates_dict['pyalgo' ].values(),
                                                               m_rates_dict['peaknet'].values())],
        )

        self.data_source = ColumnDataSource(data_source)


    def init_config(self):
        return self.data_source
