"""
Profiler module for kernel profiling.

This module provides the Profiler class, which is the main entry point
for profiling GPU kernels with different engine backends.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any

from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.engine import ComputeEngine
from kernel_scan.core.logging import get_logger
from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.specs import AcceleratorSpec, KernelSpec
from kernel_scan.core.types import EngineType

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
        accelerator_spec: Optional[AcceleratorSpec] = None,
    ):
        """
        Initialize a new Profiler instance.

        Args:
            config: Optional configuration for the profiler
            accelerator_spec: Optional accelerator specifications
        """
        self._config = config or ProfileConfig.create_default()
        self._engines: Dict[EngineType, ComputeEngine] = {}
        # Initialize with None for kernel_spec since we don't have it yet
        # We'll set it properly when we profile a specific kernel
        self._result_set = ProfileResultSet(accelerator_spec=accelerator_spec)
        self._accelerator_spec = accelerator_spec

    @property
    def config(self) -> ProfileConfig:
        """Return the profiler configuration."""
        return self._config

    @property
    def result_set(self) -> ProfileResultSet:
        """Return the profile result set."""
        return self._result_set

    @property
    def accelerator_spec(self) -> Optional[AcceleratorSpec]:
        """Return the accelerator specification."""
        return self._accelerator_spec

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
            ProfileResultSet containing the profiling results with the kernel_spec always set

        Raises:
            ValueError: If the engine type is not supported or the kernel specification is not supported
        """
        # Apply warmup iterations to config if specified
        if warmup_iterations is not None:
            self._config.warmup_iterations = warmup_iterations

        # Get or create the engine
        engine = self._get_engine(engine_type, accelerator_spec=self._accelerator_spec)

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

        # Create a new ProfileResultSet with the result and kernel_spec
        result_set = ProfileResultSet(
            results=[result],
            accelerator_spec=self._accelerator_spec,
            kernel_spec=kernel_spec,
        )

        result_set.engine_name = engine.name

        self._result_set.add_result(result)
        if self._result_set.kernel_spec is None:
            self._result_set.kernel_spec = kernel_spec

        return result_set

    def _get_engine(
        self,
        engine_type: Union[EngineType, str],
        accelerator_spec: Optional[AcceleratorSpec],
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
                    engine_type, self._config, self._accelerator_spec
                )
            elif engine_type == EngineType.MOCK:
                from kernel_scan.api.engines.mock_engine import MockEngine

                engine = MockEngine.create(
                    engine_type, self._config, self._accelerator_spec
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

    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the hardware used by this engine.

        Returns:
            Dictionary containing hardware information
        """
        if self._accelerator_specs is None:
            # If no accelerator specs have been set, return minimal info
            return {
                "name": "Unknown",
                "vendor": "Unknown",
            }

        # Return hardware info from accelerator specs
        hw_info = {
            "name": self._accelerator_specs.name,
            "memory_size_gb": self._accelerator_specs.memory_size_gb,
            "peak_memory_bandwidth_gbps": self._accelerator_specs.peak_memory_bandwidth_gbps,
        }

        # Add additional specs
        hw_info.update(self._accelerator_specs.additional_specs)

        return hw_info
