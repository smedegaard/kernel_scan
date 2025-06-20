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
        FlopsPerByte,
        GigaByte,
        GigaBytesPerSecond,
        GigaFlops,
        MegaFlops,
        Millisecond,
        # Enum for prefix constants
        Prefix,
        TeraBytePerSecond,
        TeraFlops,
        # Helper functions
        peak_performance,
    )
except ImportError as e:
    # log.error(f"Error importing kernel_scan: {e}")
    # log.error(
    #     "Make sure the kernel_scan package is properly installed or in the Python path."
    # )
    raise e


# Configure logging - MUST be imported after adding src to sys.path

# Configure logging with desired level
configure_logging(level="info")
log = get_logger(__name__)


def main():
    """Demonstrate the units module with a GPU performance analysis example."""
    print("Kernel Scan Units Module Example")
    print("================================\n")

    # 1. Creating units with different prefixes
    print("1. Creating units with different prefixes")
    print("----------------------------------------")

    # Create compute performance units
    tflops = TeraFlops(15.7)
    gflops = GigaFlops(980.0)
    mflops = MegaFlops(250.0)

    # Create bandwidth units
    gbps = GigaBytesPerSecond(900.0)
    tbps = TeraBytePerSecond(1.2)

    # Create memory size units
    memory = GigaByte(16.0)

    # Create time units
    time = Millisecond(3.5)

    # Display the units
    print(f"Compute Performance: {tflops}")
    print(f"Another Performance: {gflops}")
    print(f"Memory Bandwidth:    {gbps}")
    print(f"Memory Size:         {memory}")
    print(f"Execution Time:      {time}")
    print()

    # 2. Converting between prefixes
    print("2. Converting between prefixes")
    print("----------------------------")

    # Convert TeraFLOPS to GigaFLOPS
    tflops_to_gflops = tflops.to_giga()
    print(f"{tflops} = {tflops_to_gflops}")

    # Convert GigaFLOPS to TeraFLOPS
    gflops_to_tflops = gflops.to_tera()
    print(f"{gflops} = {gflops_to_tflops}")

    # Convert using generic with_prefix method
    tflops_to_pflops = tflops.with_prefix(Prefix.PETA)
    print(f"{tflops} = {tflops_to_pflops}")

    # Convert bytes with different prefixes
    memory_in_mb = memory.to_mega()
    print(f"{memory} = {memory_in_mb}")
    print()

    # 3. Arithmetic operations
    print("3. Arithmetic operations")
    print("-----------------------")

    # Addition
    total_flops = tflops + gflops_to_tflops
    print(f"Addition: {tflops} + {gflops_to_tflops} = {total_flops}")

    # Subtraction
    diff_flops = tflops - gflops_to_tflops
    print(f"Subtraction: {tflops} - {gflops_to_tflops} = {diff_flops}")

    # Scalar multiplication
    double_perf = tflops * 2
    print(f"Scalar multiplication: {tflops} * 2 = {double_perf}")

    # Scalar division
    half_perf = tflops / 2
    print(f"Scalar division: {tflops} / 2 = {half_perf}")
    print()

    # 4. Roofline model analysis
    print("4. Roofline model analysis")
    print("-------------------------")

    # Define hardware characteristics
    print("Hardware specifications:")
    print(f"  Peak compute:   {tflops}")
    print(f"  Peak bandwidth: {gbps}")
    print()

    # Define kernels with different arithmetic intensities
    print("Analyzing kernels with different arithmetic intensities:")

    # Create arithmetic intensity objects
    ai_values = [
        FlopsPerByte(0.1),  # Memory-bound
        FlopsPerByte(1.0),  # Balanced
        FlopsPerByte(100.0),  # Compute-bound
    ]

    # Calculate attainable performance for each kernel
    for i, ai in enumerate(ai_values):
        # Calculate attainable performance based on roofline model
        attainable = peak_performance(tflops, gbps, ai)

        # Determine if memory-bound or compute-bound
        memory_bound = (ai.base_value * gbps.base_value) < tflops.base_value
        bound_type = "memory-bound" if memory_bound else "compute-bound"

        # Display analysis
        print(f"Kernel {i + 1} (AI = {ai.format(4)}):")
        print(f"  - Status: {bound_type}")
        print(f"  - Peak attainable: {attainable}")

        # Calculate efficiency for a simulated measured performance
        # (Let's say we achieve 80% of attainable performance)
        measured = attainable.base_value * 0.8 / attainable.prefix.factor
        measured_perf = Flops(measured, attainable.prefix)
        efficiency = measured_perf.base_value / attainable.base_value

        print(f"  - Measured: {measured_perf}")
        print(f"  - Efficiency: {efficiency:.2%}")
        print()

    # 5. Calculate arithmetic intensity for a specific kernel
    print("5. Calculate arithmetic intensity for a specific kernel")
    print("----------------------------------------------------")

    # Example: Matrix multiplication kernel (M=N=K=1024)
    print("Matrix multiplication kernel (M=N=K=1024):")

    # Define computation and memory requirements
    # For MxNxK GEMM, computation is 2*M*N*K FLOPs
    # and memory access is M*K + K*N + M*N elements
    M, N, K = 1024, 1024, 1024
    flops_count = 2 * M * N * K
    memory_bytes = (M * K + K * N + M * N) * 4  # Assuming FP32 (4 bytes per element)

    # Create unit objects
    kernel_flops = Flops(flops_count)
    kernel_bytes = Byte(memory_bytes)

    # Print in more readable format
    print(f"  Computation: {kernel_flops.to_giga().format(2)} operations")
    print(f"  Memory access: {kernel_bytes.to_mega().format(2)}")

    # Calculate arithmetic intensity
    ai = flops_count / memory_bytes
    kernel_ai = FlopsPerByte(ai)
    print(f"  Arithmetic intensity: {kernel_ai.format(4)}")

    # Determine attainable performance
    attainable = peak_performance(tflops, gbps, kernel_ai)
    memory_bound = (kernel_ai.base_value * gbps.base_value) < tflops.base_value
    bound_type = "memory-bound" if memory_bound else "compute-bound"

    print(f"  Bottleneck: {bound_type}")
    print(f"  Attainable performance: {attainable}")
    print()


if __name__ == "__main__":
    main()
