"""
Base engine interface for kernel profiling.

This module defines the ComputeEngine abstract base class that all compute
engine implementations must inherit from. It also provides a factory function
for creating engine instances.
"""

import abc
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.results import ProfileResult
from kernel_scan.core.specs import KernelSpec


class EngineType(Enum):
    """Enum representing available compute engine types."""

    COMPOSABLE_KERNEL = auto()  # AMD's Composable Kernel
    MOCK = auto()  # Mock engine for testing


class ComputeEngine(abc.ABC):
    """
    Abstract base class for compute engine implementations.

    All concrete engine implementations must inherit from this class and
    implement its abstract methods.
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        """
        Initialize the compute engine.

        Args:
            config: Optional configuration for the engine
        """
        self._config = config or ProfileConfig.create_default()
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the name of the engine."""
        return self.__class__.__name__

    @property
    def config(self) -> ProfileConfig:
        """Return the engine configuration."""
        return self._config

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
    def profile(self, kernel_spec: KernelSpec) -> ProfileResult:
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

    @abc.abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the hardware used by this engine.

        Returns:
            Dictionary containing hardware information
        """
        pass

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


def create_engine(
    engine_type: Union[EngineType, str], config: Optional[ProfileConfig] = None
) -> ComputeEngine:
    """
    Factory function for creating engine instances.

    Args:
        engine_type: Type of engine to create
        config: Optional configuration for the engine

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

    # Create the appropriate engine instance
    if engine_type == EngineType.COMPOSABLE_KERNEL:
        from kernel_scan.engines.composable_kernel_engine import ComposableKernelEngine

        return ComposableKernelEngine(config)
    else:
        valid_types = [e.name for e in EngineType]
        raise ValueError(
            f"Unsupported engine type: {engine_type}. "
            f"Valid types are: {', '.join(valid_types)}"
        )
