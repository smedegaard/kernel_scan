#!/usr/bin/env python3
"""
Quick Start Example

This script demonstrates the kernel_scan library's API as shown in the README's
Quick Start section.
"""

import sys
from pathlib import Path

# Add the src directory to sys.path to import kernel_scan
# Note: We need to go up two levels since we're in examples/composable_kernel/
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    from kernel_scan import KernelSpec, Profiler, TensorSpec
    from kernel_scan.core.config import ProfilerConfigBuilder
    from kernel_scan.core.logging import configure_logging, get_logger
    from kernel_scan.operations.gemm import GemmParams
    from kernel_scan.types import DataType, EngineType, Layout, OperationType
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
    """Run the Quick Start example."""
    log.info("Kernel Scan - Quick Start Example")
    log.info("=================================")

    # Create a GEMM kernel specification
    log.info("\nCreating GEMM kernel specification...")

    kernel_spec = (
        KernelSpec.builder()
        .operation_type(OperationType.GEMM)
        .data_type(DataType.FLOAT32)
        .operation_params(
            GemmParams(
                m=1024,
                n=1024,
                k=1024,
                alpha=1.0,
                beta=0.0,
                layout_a=Layout.ROW_MAJOR,
                layout_b=Layout.ROW_MAJOR,
                layout_c=Layout.ROW_MAJOR,
            )
        )
        .inputs(
            a=TensorSpec.create_2d(1024, 1024, Layout.ROW_MAJOR, DataType.FLOAT32),
            b=TensorSpec.create_2d(1024, 1024, Layout.ROW_MAJOR, DataType.FLOAT32),
        )
        .outputs(c=TensorSpec.create_2d(1024, 1024, Layout.ROW_MAJOR, DataType.FLOAT32))
        .iterations(10)  # Reduced for quick example
        .build()
    )
    log.info("Kernel specification created successfully.")

    # Initialize profiler with kernel specification
    log.info("\nInitializing profiler...")
    # create default profiler configuration
    profiler_config = ProfilerConfigBuilder().build()
    profiler = Profiler(profiler_config, kernel_spec=kernel_spec)

    log.info("Profiling with ComposableKernelEngine...")
    try:
        result_set = profiler.profile_with_engine(
            EngineType.COMPOSABLE_KERNEL,
            warmup_iterations=2,
            output_file=Path("./results/quickstart.jsonl"),
        )

        log.info("RESULT SET")
        log.info(result_set)
        for result in result_set.results:
            log.info(f"Result: {result}")
    except Exception as e:
        log.error(f"Error during profiling: {e}")
        raise e

    log.info("Quick Start example completed!")


if __name__ == "__main__":
    main()
