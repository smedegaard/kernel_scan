"""
Core components for kernel profiling.

This package contains the core abstractions and data types used
throughout the kernel_scan library.
"""

from kernel_scan.core.types import (
    OperationType,
    Layout,
    DataType,
    TensorSpec,
    GemmParams,
    OperationParams,
    GemmOperationParams,
    OperationInputs,
    GemmInputs,
    OperationOutputs,
    GemmOutputs,
    
    # Exceptions
    KernelSpecError,
    MissingOperationTypeError,
    MissingDataTypeError,
    MissingOperationParamsError,
    MissingInputError,
    MissingOutputError,
    IncompatibleDimensionsError,
    UnsupportedOperationTypeError,
    IncompatibleLayoutError,
    IncompatibleDataTypesError,
    InvalidTensorShapeError,
    OperationParameterMismatchError,
    InsufficientWorkspaceError,
)

from kernel_scan.core.specs import (
    KernelSpec,
    KernelSpecBuilder,
)

from kernel_scan.core.results import (
    TimingData,
    ProfileResult,
    ProfileResultSet,
)

from kernel_scan.core.config import (
    ProfileConfig,
    ConfigBuilder,
)

__all__ = [
    'OperationType',
    'Layout',
    'DataType',
    'TensorSpec',
    'GemmParams',
    'OperationParams',
    'GemmOperationParams',
    'OperationInputs',
    'GemmInputs',
    'OperationOutputs',
    'GemmOutputs',
    
    # Exceptions
    'KernelSpecError',
    'MissingOperationTypeError',
    'MissingDataTypeError',
    'MissingOperationParamsError',
    'MissingInputError',
    'MissingOutputError',
    'IncompatibleDimensionsError',
    'UnsupportedOperationTypeError',
    'IncompatibleLayoutError',
    'IncompatibleDataTypesError',
    'InvalidTensorShapeError',
    'OperationParameterMismatchError',
    'InsufficientWorkspaceError',
    
    # Specs
    'KernelSpec',
    'KernelSpecBuilder',
    
    # Results
    'TimingData',
    'ProfileResult',
    'ProfileResultSet',
    
    # Config
    'ProfileConfig',
    'ConfigBuilder',
]