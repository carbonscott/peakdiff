#!/usr/bin/env python
# -*- coding: utf-8 -*-

from peakdiff.data   import DataSource
from peakdiff.viewer import Viewer

from dataclasses import dataclass

data_source_config_args = {
    "path_n_peaks_A" : "peaknet.n_peaks.msgpack",
    "path_m_rates_A" : "peaknet.m_rate.msgpack",
    "path_n_peaks_B" : "pyalgo.n_peaks.msgpack",
    "path_m_rates_B" : "pyalgo.m_rate.msgpack",
}

@dataclass
class DataSourceConfig:
    path_n_peaks_A: str
    path_m_rates_A: str
    path_n_peaks_B: str
    path_m_rates_B: str

data_source_config = DataSourceConfig(**data_source_config_args)
data_source = DataSource(data_source_config).init_config()

viewer = Viewer(data_source = data_source)

viewer.run()
