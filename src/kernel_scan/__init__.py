"""
Kernel Scan: GPU Kernel Profiling Made Simple

A Python library for profiling GPU compute kernels across different hardware
and library backends, focused on simplicity and usability.
"""

import logging

# Import core components
from kernel_scan.core.engine import EngineType
from kernel_scan.core.profiler import Profiler
from kernel_scan.core.specs import KernelSpec
from kernel_scan.core.types import DataType, Layout, OperationType, TensorSpec

# Import operation-specific components
from kernel_scan.ops import GemmParams

# Configure logging
log = logging.getLogger(__name__)

# Make key components available at the package level
__all__ = [
    # Core classes
    "Profiler",
    "KernelSpec",
    "EngineType",
    # Core types
    "OperationType",
    "DataType",
    "Layout",
    "TensorSpec",
    # Operation-specific
    "GemmParams",
]
