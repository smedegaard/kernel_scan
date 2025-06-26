#!/usr/bin/env python3
"""
Example demonstrating the use of the kernel_scan units module.

This example shows how to use the units module for GPU performance
analysis, including unit creation, conversion, and performing
roofline model calculations.
"""

import sys
from pathlib import Path

# Add the src directory to sys.path to import kernel_scan
# Note: We need to go up two levels since we're in examples/composable_kernel/
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    # Import kernel_scan modules with the new GemmScan API
    from kernel_scan.core.logging import configure_logging, get_logger
    from kernel_scan.core.units import (
        Byte,
        # Base units
        Flops,
        # Derived units
        GigaByte,
        GigaBytesPerSecond,
        GigaFlops,
        MilliSecond,
        # Enum for prefix constants
        Prefix,
        TeraFlops,
    )
except ImportError as e:
    # log.error(f"Error importing kernel_scan: {e}")
    # log.error(
    #     "Make sure the kernel_scan package is properly installed or in the Python path."
    # )
    raise e

# Configure logging with desired level
configure_logging(level="info")
log = get_logger(__name__)

# Configure logging - MUST be imported after adding src to sys.path

# Configure logging with desired level
configure_logging(level="info")
log = get_logger(__name__)


def main():
    """Demonstrate the units module with a GPU performance analysis example."""
    log.info("Kernel Scan Units Module Example")
    log.info("================================\n")

    # 1. Creating units with different prefixes
    log.info("1. Creating units with different prefixes")
    log.info("----------------------------------------")

    # Create compute performance units
    tflops = TeraFlops(15.7)
    gflops = GigaFlops(980.0)

    # Create bandwidth units
    gbps = GigaBytesPerSecond(900.0)

    # Create memory size units
    memory = GigaByte(16.0)

    # Create time units
    time = MilliSecond(3.5)

    # Display the units
    log.info(f"Compute Performance: {tflops}")
    log.info(f"Another Performance: {gflops}")
    log.info(f"Memory Bandwidth:    {gbps}")
    log.info(f"Memory Size:         {memory}")
    log.info(f"Execution Time:      {time}")

    # 2. Converting between prefixes
    log.info("2. Converting between prefixes")
    log.info("----------------------------")

    # Convert TeraFLOPS to GigaFLOPS
    tflops_to_gflops = tflops.to_giga()
    log.info(f"{tflops} = {tflops_to_gflops}")

    # Convert GigaFLOPS to TeraFLOPS
    gflops_to_tflops = gflops.to_tera()
    log.info(f"{gflops} = {gflops_to_tflops}")

    # Convert using generic with_prefix method
    tflops_to_pflops = tflops.with_prefix(Prefix.PETA)
    log.info(f"{tflops} = {tflops_to_pflops}")

    # Convert bytes with different prefixes
    memory_in_mb = memory.to_mega()
    log.info(f"{memory} = {memory_in_mb}")

    # 3. Arithmetic operations
    log.info("3. Arithmetic operations")
    log.info("-----------------------")

    # Addition
    total_flops = tflops + gflops_to_tflops
    log.info(f"Addition: {tflops} + {gflops_to_tflops} = {total_flops}")

    # Subtraction
    diff_flops = tflops - gflops_to_tflops
    log.info(f"Subtraction: {tflops} - {gflops_to_tflops} = {diff_flops}")

    # Scalar multiplication
    double_perf = tflops * 2
    log.info(f"Scalar multiplication: {tflops} * 2 = {double_perf}")

    # Scalar division
    half_perf = tflops / 2
    log.info(f"Scalar division: {tflops} / 2 = {half_perf}")

    # 4. Roofline model analysis
    log.info("4. Roofline model analysis")
    log.info("-------------------------")

    # Define hardware characteristics
    log.info("Hardware specifications:")
    log.info(f"  Peak compute:   {tflops}")
    log.info(f"  Peak bandwidth: {gbps}")

    # Define kernels with different arithmetic intensities
    log.info("Analyzing kernels with different arithmetic intensities:")

    # Example: Matrix multiplication kernel (M=N=K=1024)
    log.info("Matrix multiplication kernel (M=N=K=1024):")

    # Define computation and memory requirements
    # For MxNxK GEMM, computation is 2*M*N*K FLOPs
    # and memory access is M*K + K*N + M*N elements
    M, N, K = 1024, 1024, 1024
    flops_count = 2 * M * N * K
    memory_bytes = (M * K + K * N + M * N) * 4  # Assuming FP32 (4 bytes per element)

    # Create unit objects
    kernel_flops = Flops(flops_count)
    kernel_bytes = Byte(memory_bytes)

    # log.info in more readable format
    log.info(f"  Computation: {kernel_flops.to_giga().format(2)} operations")
    log.info(f"  Memory access: {kernel_bytes.to_mega().format(2)}")


if __name__ == "__main__":
    main()
