"""
Base engine interface for kernel profiling.

This module defines the ComputeEngine abstract base class that all compute
engine implementations must inherit from.
"""

import abc
from typing import Any, Dict, List, Optional, Union

from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.specs import  KernelSpec
from kernel_scan.core.types import EngineType


class ComputeEngine(abc.ABC):
    # Each subclass must declare its engine type
    ENGINE_TYPE: EngineType = None  # Will be overridden by subclasses

    def __init__(
        self,
        config: Optional[ProfileConfig] = None,
    ):
        """
        Initialize the compute engine.

        Args:
            config: Optional configuration for the engine
            accelerator_specs: Optional accelerator specifications for the engine
        """
        if self.ENGINE_TYPE is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define ENGINE_TYPE class attribute"
            )

        self._engine_type = self.ENGINE_TYPE
        self._config = config or ProfileConfig.create_default()
        self._initialized = False

    @property
    def engine_type(self) -> EngineType:
        """Return the engine type."""
        return self._engine_type

    @classmethod
    def create(
        cls,
        engine_type: Union[EngineType, str],
        config: Optional[ProfileConfig] = None,
    ) -> "ComputeEngine":
        """
        Factory method for creating engine instances.
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

        # Validate that this class can handle the requested engine type
        if engine_type != cls.ENGINE_TYPE:
            raise ValueError(
                f"{cls.__name__} cannot create engine of type {engine_type}. "
                f"Expected {cls.ENGINE_TYPE}"
            )

        return cls(config)

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
