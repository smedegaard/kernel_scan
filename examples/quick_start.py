#!/usr/bin/env python3
"""
Quick Start Example

This script demonstrates the kernel_scan library's API as shown in the README's
Quick Start section.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Add the src directory to sys.path to import kernel_scan
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    # Import kernel_scan modules
    from kernel_scan import (
        DataType,
        EngineType,
        KernelSpec,
        Layout,
        OperationType,
        Profiler,
    )
    from kernel_scan.core import AcceleratorSpecs
    from kernel_scan.core.types import TensorSpec
    from kernel_scan.ops import GemmParams
except ImportError as e:
    log.error(f"Error importing kernel_scan: {e}")
    log.error(
        "Make sure the kernel_scan package is properly installed or in the Python path."
    )
    sys.exit(1)


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

    # Detect the accelerator hardware
    # log.info("\nDetecting GPU hardware...")
    # try:
    #     # Auto-detect hardware
    #     accelerator = AcceleratorSpecs.detect_hardware()
    #     log.info(f"Detected GPU: {accelerator.name}")
    #     log.info(f"Memory: {accelerator.memory_size_gb} GB")
    #     log.info(
    #         f"Peak Memory Bandwidth: {accelerator.peak_memory_bandwidth_gbps} GB/s"
    #     )

    #     # Display peak performance for different precisions
    #     for dtype, tflops in accelerator.peak_performance.items():
    #         log.info(f"Peak Performance ({dtype.name}): {tflops} TFLOPS")

    #     # Show any additional specs
    #     if accelerator.additional_specs:
    #         log.info("Additional Hardware Specifications:")
    #         for key, value in accelerator.additional_specs.items():
    #             log.info(f"  {key}: {value}")
    # except Exception as e:
    #     log.warning(f"Could not detect GPU hardware: {e}")
    #     log.warning("Will continue with default/unknown hardware specifications.")
    #     accelerator = None

    # Profile with a specific engine
    log.info("\nInitializing profiler...")
    profiler = Profiler()

    log.info("Profiling with ComposableKernelEngine...")
    try:
        # Pass the detected accelerator specs to the profiler
        result_set = profiler.profile_with_engine(
            kernel_spec,
            EngineType.COMPOSABLE_KERNEL,
            warmup_iterations=2,
            output_file="./results/quickstart.jsonl",
        )

        print(f"number of results: {len(result_set.results)}")

    except Exception as e:
        log.error(f"Error during profiling: {e}")
        raise e

    log.info("\nQuick Start example completed!")


if __name__ == "__main__":
    main()
