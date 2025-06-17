"""
Core components for kernel profiling.

This package contains the core abstractions and data types used
throughout the kernel_scan library.
"""

from kernel_scan.core.accelerator import AcceleratorSpecs
from kernel_scan.core.config import (
    ConfigBuilder,
    ProfileConfig,
)
from kernel_scan.core.results import (
    ProfileResult,
    ProfileResultSet,
    TimingData,
)
from kernel_scan.core.specs import (
    KernelSpec,
    KernelSpecBuilder,
)
from kernel_scan.core.types import (
    DataType,
    GemmInputs,
    GemmOperationParams,
    GemmOutputs,
    GemmParams,
    IncompatibleDataTypesError,
    IncompatibleDimensionsError,
    IncompatibleLayoutError,
    InsufficientWorkspaceError,
    InvalidTensorShapeError,
    # Exceptions
    KernelSpecError,
    Layout,
    MissingDataTypeError,
    MissingInputError,
    MissingOperationParamsError,
    MissingOperationTypeError,
    MissingOutputError,
    OperationInputs,
    OperationOutputs,
    OperationParameterMismatchError,
    OperationParams,
    OperationType,
    TensorSpec,
    UnsupportedOperationTypeError,
)

__all__ = [
    "OperationType",
    "Layout",
    "DataType",
    "TensorSpec",
    "GemmParams",
    "OperationParams",
    "GemmOperationParams",
    "OperationInputs",
    "GemmInputs",
    "OperationOutputs",
    "GemmOutputs",
    # Exceptions
    "KernelSpecError",
    "MissingOperationTypeError",
    "MissingDataTypeError",
    "MissingOperationParamsError",
    "MissingInputError",
    "MissingOutputError",
    "IncompatibleDimensionsError",
    "UnsupportedOperationTypeError",
    "IncompatibleLayoutError",
    "IncompatibleDataTypesError",
    "InvalidTensorShapeError",
    "OperationParameterMismatchError",
    "InsufficientWorkspaceError",
    # Specs
    "KernelSpec",
    "KernelSpecBuilder",
    # Results
    "TimingData",
    "ProfileResult",
    "ProfileResultSet",
    # Config
    "ProfileConfig",
    "ConfigBuilder",
    # Accelerator
    "AcceleratorSpecs",
]
