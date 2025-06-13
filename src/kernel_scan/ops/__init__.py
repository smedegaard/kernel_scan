"""
Operation implementations for kernel profiling.

This package contains the implementations of various operations
that can be profiled using the kernel_scan library.
"""

from kernel_scan.ops.gemm import (
    validate_gemm_operation,
    calculate_gemm_flops,
    create_random_gemm_inputs,
    verify_gemm_result,
)
from kernel_scan.core.types import (
    GemmParams,
    GemmInputs,
    GemmOutputs,
)

__all__ = [
    # GEMM operations
    'GemmParams',
    'GemmInputs',
    'GemmOutputs',
    'validate_gemm_operation',
    'calculate_gemm_flops',
    'create_random_gemm_inputs',
    'verify_gemm_result',
]