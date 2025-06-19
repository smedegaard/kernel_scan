"""
Kernel Scan: GPU Kernel Profiling Made Simple

A Python library for profiling GPU compute kernels across different hardware
and library backends, focused on simplicity and usability.
"""

import logging

import kernel_scan.core.types as types

# Re-export key API classes for direct import from kernel_scan
from kernel_scan.api import Profiler, visualization

# Import engine types that users will need
from kernel_scan.core.engine import EngineType
from kernel_scan.core.specs import (
    AcceleratorSpec,
    KernelSpec,
    KernelSpecBuilder,
    TensorSpec,
)

# Re-export types directly for backward compatibility
from kernel_scan.core.types import (
    DataType,
    Layout,
    OperationInputs,
    OperationOutputs,
    OperationParams,
    OperationType,
)

# Configure logging
log = logging.getLogger(__name__)

# Make key components available at the package level
__all__ = [
    "types",
    "visualization",
    # Main API classes
    "Profiler",
    "KernelSpecBuilder",
    # Specs
    "TensorSpec",
    "AcceleratorSpec",
    "KernelSpec",
    "EngineType",
    # Re-exported types
    "DataType",
    "Layout",
    "OperationType",
    "OperationInputs",
    "OperationOutputs",
    "OperationParams",
]
