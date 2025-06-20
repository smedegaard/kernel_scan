#!/usr/bin/env python3
"""
GEMM Performance Scanner

This script demonstrates the kernel_scan library's API for profiling GEMM operations
across different matrix sizes. It follows the Quick Start pattern from the README
while scanning GEMM performance for N=K (powers of 2 from 64 to 16384) with
M=1,2,4,...,256 using multiple data types (FP16, BF16, FP32, INT8).
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Add the src directory to sys.path to import kernel_scan
# Note: We need to go up two levels since we're in examples/composable_kernel/
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    # Import kernel_scan modules with the new GemmScan API
    from kernel_scan.operations.gemm import GemmScan
    from kernel_scan.types import DataType, EngineType
    from kernel_scan.visualization import generate_gemm_roofline_plots_by_group
except ImportError as e:
    log.error(f"Error importing kernel_scan: {e}")
    log.error(
        "Make sure the kernel_scan package is properly installed or in the Python path."
    )
    raise e


def main():
    """Run the GEMM performance scan using the new GemmScan API."""
    log.info("Starting GEMM performance scan...")

    # Initialize the GemmScan object with default parameters
    scan = GemmScan()
    # Configure and run the scan using the fluent API
    results = (
        scan.with_data_types([DataType.FLOAT16])
        .for_m_values([1, 2, 4])  # M grows through these values
        .growing_with_respect_to("M")  # M is the growing dimension
        .with_n_equals(lambda m: m * 256)  # N = M * 256
        .with_k_equals(lambda m: m * 256)  # K = M * 256
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
            for result_set in data_type_results:
                figures = generate_gemm_roofline_plots_by_group(result_set)
                for group, fig in figures.items():
                    plot_file = scan._plots_dir / f"{data_type}_{group}.png"
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
