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
    from kernel_scan.types import DataType, EngineType, Layout
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
    (
        scan.with_engine_type(EngineType.COMPOSABLE_KERNEL)
        .with_data_types(
            [
                DataType.FLOAT16,
                DataType.BFLOAT16,
            ]
        )
        .for_n_values([2**n for n in range(6, 15)])  # powers of 2 from 2⁶ to 2¹⁴
        .for_m_values([2**n for n in range(0, 9)])  # powers of 2 from 2⁰ to 2⁸
        .with_k_equals_n()  # Configure the scan to use K = N for all test cases
        .iterations(10)
        .warmup(5)
        .with_layouts(
            layout_a=Layout.ROW_MAJOR,
            layout_b=Layout.ROW_MAJOR,
            layout_c=Layout.ROW_MAJOR,
        )
        .with_scaling(alpha=1.0, beta=0.0)  # Set scaling factors
        .output_to(Path("results"))
        .run()  # Execute the scan
    )

    log.info(f"\nScan completed!\n Results saved to: {scan._base_output_dir}")


if __name__ == "__main__":
    main()
