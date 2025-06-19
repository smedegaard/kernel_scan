"""
Kernel Scan: GPU Kernel Profiling Made Simple

A Python library for profiling GPU compute kernels across different hardware
and library backends, focused on simplicity and usability.
"""

import logging
import sys

# Import API components
from kernel_scan.api import Profiler, engines, operations, visualization

# Import core types and specs
from kernel_scan.core import types
from kernel_scan.core.specs import (
    AcceleratorSpec,
    KernelSpec,
    KernelSpecBuilder,
    TensorSpec,
)

# Create aliases for commonly used modules to allow for more intuitive imports
sys.modules["kernel_scan.operations"] = operations
sys.modules["kernel_scan.types"] = types
sys.modules["kernel_scan.visualization"] = visualization

# Also make the specific operation modules available directly
# For example, this makes kernel_scan.api.operations.gemm available as kernel_scan.operations.gemm
if hasattr(operations, "gemm"):
    sys.modules["kernel_scan.operations.gemm"] = operations.gemm

# Make key components available at the package level
__all__ = [
    # API modules
    "Profiler",
    "operations",
    "engines",
    "visualization",
    # Core modules
    "types",
    # Core specs
    "TensorSpec",
    "AcceleratorSpec",
    "KernelSpec",
    "KernelSpecBuilder",
    "EngineType",
]

# Configure logging
log = logging.getLogger(__name__)
