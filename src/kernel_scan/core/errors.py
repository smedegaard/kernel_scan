"""Exceptions for kernel specification errors.

This module defines custom exceptions that are raised when validating or processing
kernel specifications. These exceptions help identify specific issues with kernel
configurations, tensor specifications, and operation parameters.
"""

from kernel_scan.core.types import OperationType

###################################################################################
#                                                                                 #
#      IN ORDER TO AVOID CIRCULAR IMPORTS                                         #
#      THIS FILE SHOULD  ONLY EVER IMPORT from `core.types`                       #
#                                                                                 #
###################################################################################


class KernelSpecError(Exception):
    """Base exception for kernel specification errors."""

    pass


class MissingOperationTypeError(KernelSpecError):
    """Raised when operation type is missing."""

    def __init__(self):
        super().__init__("Missing operation type")


class MissingDataTypeError(KernelSpecError):
    """Raised when data type is missing."""

    def __init__(self):
        super().__init__("Missing data type")


class MissingOperationParamsError(KernelSpecError):
    """Raised when operation parameters are missing."""

    def __init__(self):
        super().__init__("Missing operation parameters")


class MissingInputError(KernelSpecError):
    """Raised when an input tensor is missing."""

    def __init__(self, input_name: str):
        super().__init__(f"Missing input tensor: {input_name}")


class MissingOutputError(KernelSpecError):
    """Raised when an output tensor is missing."""

    def __init__(self, output_name: str):
        super().__init__(f"Missing output tensor: {output_name}")


class IncompatibleDimensionsError(KernelSpecError):
    """Raised when tensor dimensions are incompatible."""

    def __init__(self, message: str):
        super().__init__(f"Incompatible dimensions: {message}")


class UnsupportedOperationTypeError(KernelSpecError):
    """Raised when an operation type is not supported."""

    def __init__(self, op_type: OperationType):
        super().__init__(f"Unsupported operation type: {op_type}")


class IncompatibleLayoutError(KernelSpecError):
    """Raised when tensor layouts are incompatible."""

    def __init__(self, message: str):
        super().__init__(f"Incompatible layout: {message}")


class IncompatibleDataTypesError(KernelSpecError):
    """Raised when tensor data types are incompatible."""

    def __init__(self, message: str):
        super().__init__(f"Incompatible data types: {message}")


class InvalidTensorShapeError(KernelSpecError):
    """Raised when a tensor shape is invalid."""

    def __init__(self, message: str):
        super().__init__(f"Invalid tensor shape: {message}")


class OperationParameterMismatchError(KernelSpecError):
    """Raised when operation parameters don't match inputs/outputs."""

    def __init__(self, message: str):
        super().__init__(f"Operation parameter mismatch: {message}")


class InsufficientWorkspaceError(KernelSpecError):
    """Raised when workspace size is insufficient."""

    def __init__(self, required: int, provided: int):
        super().__init__(
            f"Workspace size insufficient: required {required}, provided {provided}"
        )
