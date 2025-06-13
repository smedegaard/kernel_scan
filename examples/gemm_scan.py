#!/usr/bin/env python3
"""
GEMM Performance Scanner

This script demonstrates the kernel_scan library's API for profiling GEMM operations
across different matrix sizes. It follows the Quick Start pattern from the README
while scanning GEMM performance for N=K (powers of 2 from 64 to 16384) with
M=1,2,4,...,256 using multiple data types (FP16, BF16, FP32, INT8).
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to sys.path to import kernel_scan
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    # Import kernel_scan modules like in the Quick Start example
    # Since Profiler class is not implemented yet
    from kernel_scan.core.config import ProfileConfig
    from kernel_scan.core.specs import KernelSpec
    from kernel_scan.core.types import DataType, Layout, OperationType, TensorSpec
    from kernel_scan.engines.composable_kernel_engine import ComposableKernelEngine
    from kernel_scan.ops import GemmParams
except ImportError as e:
    print(f"Error importing kernel_scan: {e}")
    print(
        "Make sure the kernel_scan package is properly installed or in the Python path."
    )
    sys.exit(1)


def main():
    """Run the GEMM performance scan using the Quick Start pattern."""
    # Configuration settings
    output_dir = "results"
    warmup_iterations = 5
    iterations = 10

    # Data types to scan
    data_types = [
        DataType.FLOAT16,  # FP16
        DataType.BFLOAT16,  # BF16
        DataType.FLOAT32,  # FP32
        DataType.INT8,  # INT8
    ]

    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"{output_dir}/gemm_scan_{timestamp}"

    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)

    # Matrix dimensions to scan
    # N=K values (powers of 2 from 64 to 16384)
    nk_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # M values (powers of 2 from 1 to 256)
    m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Since Profiler class is not implemented yet, we'll use the engine directly
    # In the future, this would use the Profiler API as shown in Quick Start
    config = ProfileConfig()
    config.warmup_iterations = warmup_iterations
    config.output_dir = base_output_dir

    # Initialize the engine
    engine = ComposableKernelEngine(config)
    engine.initialize()

    # Print scan configuration
    print("Starting GEMM performance scan...")
    print(f"Data types: {[dt.name for dt in data_types]}")
    print(f"N=K values: {nk_values}")
    print(f"M values: {m_values}")
    print(f"Output directory: {base_output_dir}")
    print("")

    # Counter for progress
    total_tests = len(data_types) * len(nk_values) * len(m_values)
    current_test = 0

    # For storing all results
    all_results = []

    # Main scanning loop
    for data_type in data_types:
        # Create subdirectory for each data type
        dtype_output_dir = f"{base_output_dir}/{data_type.name}"
        os.makedirs(dtype_output_dir, exist_ok=True)

        print(f"\nScanning with data type: {data_type.name}")

        for nk in nk_values:
            # Create subdirectory for each N=K value
            nk_output_dir = f"{dtype_output_dir}/NK{nk}"
            os.makedirs(nk_output_dir, exist_ok=True)

            for m in m_values:
                current_test += 1
                n = nk
                k = nk

                # Output file for this configuration
                output_file = f"{nk_output_dir}/gemm_M{m}_N{n}_K{k}_{data_type.name.lower()}.jsonl"

                print(
                    f"[{current_test}/{total_tests}] Testing {data_type.name} M={m}, N={n}, K={k}"
                )

                # Create a GEMM kernel specification like in the Quick Start example
                kernel_spec = (
                    KernelSpec.builder()
                    .operation_type(OperationType.GEMM)
                    .data_type(data_type)
                    .operation_params(
                        GemmParams(
                            m=m,
                            n=n,
                            k=k,
                            alpha=1.0,
                            beta=0.0,
                            layout_a=Layout.ROW_MAJOR,
                            layout_b=Layout.ROW_MAJOR,
                            layout_c=Layout.ROW_MAJOR,
                        )
                    )
                    .inputs(
                        a=TensorSpec.create_2d(m, k, Layout.ROW_MAJOR, data_type),
                        b=TensorSpec.create_2d(k, n, Layout.ROW_MAJOR, data_type),
                    )
                    .outputs(c=TensorSpec.create_2d(m, n, Layout.ROW_MAJOR, data_type))
                    .iterations(iterations)
                    .name(f"GEMM_M{m}_N{n}_K{k}_{data_type.name}")
                    .build()
                )

                try:
                    # In the future, this would use:
                    # profiler = Profiler()
                    # result = profiler.profile_with_engine(
                    #     kernel_spec,
                    #     EngineType.COMPOSABLE_KERNEL,
                    #     warmup_iterations=warmup_iterations
                    # )

                    # For now, use the engine directly
                    result = engine.profile(kernel_spec, output_file=output_file)
                    all_results.append(result)

                    # Print results like in the Quick Start example
                    print(
                        f"  Execution time: {result.timing.avg_kernel_time_ms:.3f} ms"
                    )
                    print(f"  Throughput: {result.metrics.get('gflops', 0):.2f} GFLOPS")
                    print(f"  Results saved to: {output_file}")

                except Exception as e:
                    print(f"  ✗ Failed for {data_type.name} M={m}, N={n}, K={k}: {e}")

    print("\nScan completed!")


if __name__ == "__main__":
    main()
