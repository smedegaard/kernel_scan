#!/usr/bin/env python3
"""
GEMM Performance Scanner

This script demonstrates the kernel_scan library's API for profiling GEMM operations
across different matrix sizes. It follows the Quick Start pattern from the README
while scanning GEMM performance for N=K (powers of 2 from 64 to 16384) with
M=1,2,4,...,256 using multiple data types (FP16, BF16, FP32, INT8).
"""

import sys
from pathlib import Path

# Add the src directory to sys.path to import kernel_scan
# Note: We need to go up two levels since we're in examples/composable_kernel/
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))


try:
    # Import kernel_scan modules with the new GemmScan API
    # Configure logging - MUST be imported after adding src to sys.path
    from kernel_scan.core.logging import configure_logging, get_logger
    from kernel_scan.operations.gemm import GemmScan
    from kernel_scan.types import DataType, EngineType
    from kernel_scan.visualization import generate_gemm_roofline_plots_by_data_type
except ImportError as e:
    # log.error(f"Error importing kernel_scan: {e}")
    # log.error(
    #     "Make sure the kernel_scan package is properly installed or in the Python path."
    # )
    raise e

# Configure logging with desired level
configure_logging(level="info")
log = get_logger(__name__)


def main():
    """Run the GEMM performance scan using the new GemmScan API."""
    log.info("Starting GEMM performance scan...")

    # Initialize the GemmScan object with default parameters
    scan = GemmScan()
    # Configure and run the scan using the fluent API
    results = (
        scan.with_data_types(
            [DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16, DataType.INT8]
        )
        .for_m_values([2**n for n in range(0, 9)])  # [1, 2, 4, 8, 16, 32, 64, 128, 256]
        .for_n_values(
            [2**n for n in range(6, 15)]
        )  # [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        .with_k_equals_n()  # This sets K = N for all test cases (enables Cartesian product mode)
        .with_engine_type(EngineType.COMPOSABLE_KERNEL)
        .iterations(10)
        .warmup(5)
        .run()
    )

    # Generate and save plots
    for data_type, data_type_results in results.items():
        if not data_type_results:
            log.warning(f"No results for {data_type}, skipping plots")
            continue

        try:
            log.info(f"Generating roofline plots for {data_type}...")
            data_type_enum = (
                DataType[data_type] if isinstance(data_type, str) else data_type
            )

            figures = generate_gemm_roofline_plots_by_data_type(
                result_sets=data_type_results, data_type=data_type_enum
            )

            for precision_name, fig in figures.items():
                plot_file = scan._plots_dir / f"{precision_name}.png"
                fig.write_image(plot_file)
                log.info(f" 🖼️ Plot saved to: {plot_file}")
        except Exception as e:
            log.error(f"Error generating plots for {data_type}: {e}")
            raise e

    log.info(
        f"\nScan completed!\n Results saved to: {scan._base_output_dir}. Plots can be found in {scan._plots_dir}."
    )


if __name__ == "__main__":
    main()
