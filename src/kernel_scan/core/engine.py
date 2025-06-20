"""
Base engine interface for kernel profiling.

This module defines the ComputeEngine abstract base class that all compute
engine implementations must inherit from.
"""

import abc
from typing import Any, Dict, List, Optional, Union

from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.specs import AcceleratorSpec, KernelSpec
from kernel_scan.core.types import EngineType


class ComputeEngine(abc.ABC):
    """
    Abstract base class for compute engine implementations.

    All concrete engine implementations must inherit from this class and
    implement its abstract methods.
    """

    def __init__(
        self,
        config: Optional[ProfileConfig] = None,
        accelerator_specs: Optional[AcceleratorSpec] = None,
    ):
        """
        Initialize the compute engine.

        Args:
            config: Optional configuration for the engine
            accelerator_specs: Optional accelerator specifications for the engine
        """
        self._config = config or ProfileConfig.create_default()
        self._accelerator_specs = accelerator_specs
        self._initialized = False

    @classmethod
    @abc.abstractmethod
    def create(
        cls,
        engine_type: Union[EngineType, str],
        config: Optional[ProfileConfig] = None,
        accelerator_specs: Optional[AcceleratorSpec] = None,
    ) -> "ComputeEngine":
        """
        Factory method for creating engine instances.

        Args:
            engine_type: Type of engine to create
            config: Optional configuration for the engine
            accelerator_specs: Optional hardware specifications for the accelerator

        Returns:
            ComputeEngine instance

        Raises:
            ImportError: If the required engine implementation is not available
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

        raise NotImplementedError(
            f"Engine type {engine_type} is not supported by this engine implementation."
        )

    @property
    def name(self) -> str:
        """Return the name of the engine."""
        return self.__class__.__name__

    @property
    def config(self) -> ProfileConfig:
        """Return the engine configuration."""
        return self._config

    @property
    def accelerator_specs(self) -> AcceleratorSpec:
        """Return the accelerator specifications for the engine."""
        return self._accelerator_specs

    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the engine and any required resources.

        Returns:
            True if initialization was successful
        """
        pass

    @abc.abstractmethod
    def is_supported(self, kernel_spec: KernelSpec) -> bool:
        """
        Check if the engine supports the given kernel specification.

        Args:
            kernel_spec: The kernel specification to check

        Returns:
            True if the kernel specification is supported
        """
        pass

    @abc.abstractmethod
    def profile(self, kernel_spec: KernelSpec) -> ProfileResultSet:
        """
        Profile the given kernel specification.

        Args:
            kernel_spec: The kernel specification to profile

        Returns:
            ProfileResult containing the profiling results

        Raises:
            ValueError: If the kernel specification is not supported
        """
        pass

    @abc.abstractmethod
    def get_available_kernels(self) -> List[Dict[str, Any]]:
        """
        Get a list of available kernels supported by this engine.

        Returns:
            List of dictionaries describing available kernels
        """
        pass

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

    def shutdown(self) -> None:
        """Release resources used by the engine."""
        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
