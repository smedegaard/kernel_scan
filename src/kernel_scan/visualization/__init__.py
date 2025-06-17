"""
Visualization module for kernel profiling results.

This module provides functions and classes for visualizing kernel profiling
results using common plotting libraries.
"""

import logging

from kernel_scan.core.accelerator import AcceleratorSpecs
from kernel_scan.visualization.plots import (
    generate_gemm_roofline_plots_by_group,
)

log = logging.getLogger(__name__)

__all__ = [
    "plot_roofline",
    "generate_gemm_roofline_plots_by_group",
    "AcceleratorSpecs",
]
