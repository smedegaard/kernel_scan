"""
Profiler module for kernel profiling.

This module provides the Profiler class, which is the main entry point
for profiling GPU kernels with different engine backends.
"""

from pathlib import Path
from typing import Dict, Optional, Union

from kernel_scan.api.operations.gemm import GemmParams
from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.engine import ComputeEngine
from kernel_scan.core.logging import get_logger
from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.specs import AcceleratorSpec, KernelSpec
from kernel_scan.core.types import EngineType, Layout, OperationType

log = get_logger(__name__)


class Profiler:
    """
    Main profiler class for kernel profiling.

    This class provides a simple interface for profiling GPU kernels with
    different engine backends.
    """

    def __init__(
        self,
        config: Optional[ProfileConfig] = None,
        accelerator_specs: Optional[AcceleratorSpec] = None,
    ):
        """
        Initialize a new Profiler instance.

        Args:
            config: Optional configuration for the profiler
        """
        self._config = config or ProfileConfig.create_default()
        self._engines: Dict[EngineType, ComputeEngine] = {}
        self._result_set = ProfileResultSet()
        self._accelerator_specs = accelerator_specs

    @property
    def config(self) -> ProfileConfig:
        """Return the profiler configuration."""
        return self._config

    @property
    def result_set(self) -> ProfileResultSet:
        """Return the profile result set."""
        return self._result_set

    def profile_with_engine(
        self,
        kernel_spec: KernelSpec,
        engine_type: Union[EngineType, str],
        warmup_iterations: Optional[int] = None,
        output_file: Optional[Path] = None,
    ) -> ProfileResultSet:
        """
        Profile a kernel with a specific engine.

        Args:
            kernel_spec: The kernel specification to profile
            engine_type: The type of engine to use
            warmup_iterations: Optional number of warmup iterations to run
            output_file: Optional file to save the results to

        Returns:
            ProfileResult containing the profiling results

        Raises:
            ValueError: If the engine type is not supported or the kernel specification is not supported
        """
        # Apply warmup iterations to config if specified
        if warmup_iterations is not None:
            self._config.warmup_iterations = warmup_iterations

        # Get or create the engine
        engine = self._get_engine(
            engine_type, accelerator_specs=self._accelerator_specs
        )

        # Check if the kernel is supported
        if not engine.is_supported(kernel_spec):
            raise ValueError(f"Kernel specification not supported by {engine.name}")

        # Create parent directory for output_file if it doesn't exist
        if type(output_file) is str:
            output_file = Path(output_file)
        if output_file is not None:
            output_dir = output_file.parent
            if not output_dir.exists():
                log.warning(
                    f"Output directory does not exist, creating it now: {output_dir}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

        # Profile the kernel
        result = engine.profile(kernel_spec, output_file=output_file)

        # Save the result to the result set
        self._result_set.add_result(result)

        # Return the result
        return result

    def profile_gemm(
        self,
        m: int,
        n: int,
        k: int,
        data_type,
        engine_type: Union[EngineType, str],
        output_file: str,
        **kwargs,
    ) -> ProfileResultSet:
        """
        Profile a GEMM operation with simplified parameters.

        This is a convenience method for profiling GEMM operations without
        having to create a full KernelSpec object.

        Args:
            m: Number of rows in matrices A and C
            n: Number of columns in matrices B and C
            k: Number of columns in matrix A / rows in matrix B
            data_type: Data type for the operation
            engine_type: The type of engine to use
            **kwargs: Additional parameters for the GEMM operation

        Returns:
            ProfileResult containing the profiling results

        Raises:
            ValueError: If the engine type is not supported or the kernel specification is not supported
        """
        from kernel_scan.core.specs import TensorSpec

        # Extract additional parameters
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 0.0)
        layout_a = kwargs.get("layout_a", Layout.ROW_MAJOR)
        layout_b = kwargs.get("layout_b", Layout.ROW_MAJOR)
        layout_c = kwargs.get("layout_c", Layout.ROW_MAJOR)
        iterations = kwargs.get("iterations", 100)
        warmup_iterations = kwargs.get("warmup_iterations", 10)
        name = kwargs.get("name", f"GEMM_M{m}_N{n}_K{k}")

        # Create a GEMM kernel specification
        kernel_spec = (
            KernelSpec.builder()
            .operation_type(OperationType.GEMM)
            .data_type(data_type)
            .operation_params(
                GemmParams(
                    m=m,
                    n=n,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    layout_a=layout_a,
                    layout_b=layout_b,
                    layout_c=layout_c,
                )
            )
            .inputs(
                a=TensorSpec.create_2d(m, k, layout_a, data_type),
                b=TensorSpec.create_2d(k, n, layout_b, data_type),
            )
            .outputs(c=TensorSpec.create_2d(m, n, layout_c, data_type))
            .iterations(iterations)
            .name(name)
            .build()
        )

        # Profile with the specified engine
        return self.profile_with_engine(
            kernel_spec, engine_type, warmup_iterations, output_file
        )

    def _get_engine(
        self,
        engine_type: Union[EngineType, str],
        accelerator_specs: Optional[AcceleratorSpec],
    ) -> ComputeEngine:
        """
        Get or create an engine instance.

        Args:
            engine_type: The type of engine to get or create

        Returns:
            ComputeEngine instance

        Raises:
            ValueError: If the engine type is not supported
        """
        # Convert string to enum if needed
        if isinstance(engine_type, str):
            try:
                engine_type = next(
                    e
                    for e in EngineType
                    if e.name.lower() == engine_type.lower().replace(" ", "_")
                )
            except StopIteration:
                valid_types = [e.name.lower() for e in EngineType]
                raise ValueError(
                    f"Unsupported engine type: {engine_type}. "
                    f"Valid types are: {', '.join(valid_types)}"
                )

        # Create the engine if it doesn't exist
        if engine_type not in self._engines:
            # Import the appropriate engine implementation based on engine_type
            if engine_type == EngineType.COMPOSABLE_KERNEL:
                from kernel_scan.api.engines.composable_kernel_engine import (
                    ComposableKernelEngine,
                )

                engine = ComposableKernelEngine.create(
                    engine_type, self._config, self._accelerator_specs
                )
            elif engine_type == EngineType.MOCK:
                from kernel_scan.api.engines.mock_engine import MockEngine

                engine = MockEngine.create(
                    engine_type, self._config, self._accelerator_specs
                )
            else:
                raise ValueError(f"Unsupported engine type: {engine_type}")

            engine.initialize()
            self._engines[engine_type] = engine

        return self._engines[engine_type]

    def __del__(self):
        """Clean up resources when the profiler is deleted."""
        for engine in self._engines.values():
            engine.shutdown()
        self._engines.clear()
