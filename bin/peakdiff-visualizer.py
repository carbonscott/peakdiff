#!/usr/bin/env python
# -*- coding: utf-8 -*-

from peakdiff.data   import CXIPeakDiff
from peakdiff.viewer import CXIPeakDiffViewer

from dataclasses import dataclass

path_cxi_0 = 'cxic00318.aggregated.cxi'
path_cxi_1 = 'inference_results/peaknet.cxic00318_0123.cxi'

cxi_peakdiff = CXIPeakDiff(path_cxi_0, path_cxi_1)

viewer = CXIPeakDiffViewer(cxi_peakdiff = cxi_peakdiff)

viewer.run()
