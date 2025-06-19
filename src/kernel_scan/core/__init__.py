"""
Core components for kernel profiling.

This package contains the core abstractions and data types used
throughout the kernel_scan library.
"""

# Import types module to make it accessible
from kernel_scan.core import types

# Import specs for convenience
from kernel_scan.core.specs import (
    AcceleratorSpec,
    KernelSpec,
    KernelSpecBuilder,
    TensorSpec,
)
