"""
Operation implementations for kernel profiling.

This package contains the implementations of various operations
that can be profiled using the kernel_scan library.
"""

# Import the gemm module to make it accessible
from kernel_scan.api.operations import gemm

__all__ = ["gemm"]
