"""
Core data types for kernel profiling.

This module defines the fundamental data types and structures used
throughout the kernel_scan library for GPU kernel profiling.
"""

import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import polars as pl


class OperationType(Enum):
    """Represents the type of compute operation to be profiled."""

    GEMM = auto()  # General Matrix Multiplication
    # Future operations can be added here


class Layout(Enum):
    """Represents the memory layout of tensors."""

    ROW_MAJOR = auto()
    COLUMN_MAJOR = auto()


class DataType(Enum):
    """Represents the data type used in compute operations."""

    FLOAT32 = auto()  # 32-bit floating point (single precision)
    FLOAT16 = auto()  # 16-bit floating point (half precision)
    BFLOAT16 = auto()  # 16-bit brain floating point (used in AI accelerators)
    FLOAT64 = auto()  # 64-bit floating point (double precision)
    INT8 = auto()  # 8-bit integer (signed)
    UINT8 = auto()  # 8-bit integer (unsigned)
    INT16 = auto()  # 16-bit integer (signed)
    INT32 = auto()  # 32-bit integer (signed)
    INT64 = auto()  # 64-bit integer (signed)
    INT4 = auto()  # 4-bit integer (signed) - used in AI accelerators
    BOOL = auto()  # 8-bit boolean (0 or 1)

    @classmethod
    def get_polars_dtype(cls, dtype: "DataType") -> str:
        """Convert a DataType enum to polars dtype string."""
        mapping = {
            cls.FLOAT32: "f32",
            cls.FLOAT16: "f16",
            cls.BFLOAT16: "f32",  # Polars doesn't support bfloat16 natively
            cls.FLOAT64: "f64",
            cls.INT8: "i8",
            cls.UINT8: "u8",
            cls.INT16: "i16",
            cls.INT32: "i32",
            cls.INT64: "i64",
            cls.INT4: "i8",  # Polars doesn't support i4 natively
            cls.BOOL: "bool",
        }
        return mapping.get(dtype, "f32")

    @classmethod
    def from_string(cls, format_str: str) -> "DataType":
        """Convert string to DataType enum."""
        format_map = {
            "fp64": cls.FLOAT64,
            "float64": cls.FLOAT64,
            "double": cls.FLOAT64,
            "fp32": cls.FLOAT32,
            "float32": cls.FLOAT32,
            "float": cls.FLOAT32,
            "fp16": cls.FLOAT16,
            "float16": cls.FLOAT16,
            "half": cls.FLOAT16,
            "bf16": cls.BFLOAT16,
            "bfloat16": cls.BFLOAT16,
            "int8": cls.INT8,
            "i8": cls.INT8,
            "uint8": cls.UINT8,
            "u8": cls.UINT8,
            "int16": cls.INT16,
            "i16": cls.INT16,
            "int32": cls.INT32,
            "i32": cls.INT32,
            "int64": cls.INT64,
            "i64": cls.INT64,
            "int4": cls.INT4,
            "i4": cls.INT4,
            "bool": cls.BOOL,
            "boolean": cls.BOOL,
        }

        # Try direct lookup
        if isinstance(format_str, str):
            key = format_str.lower()
            if key in format_map:
                return format_map[key]

        # Try to match enum name
        try:
            return cls[format_str.upper()]
        except (KeyError, AttributeError):
            raise ValueError(f"Unknown data type: {format_str}")


@dataclass
class TensorSpec:
    """
    Specification for a tensor (input or output) used in compute operations.

    Attributes:
        dimensions: List of dimensions (e.g., [batch, height, width, channels])
        layout: Memory layout of the tensor
        data_type: Data type of the tensor elements
    """

    dimensions: List[int]
    layout: Layout
    data_type: DataType

    @classmethod
    def create_2d(
        cls, rows: int, cols: int, layout: Layout, data_type: DataType
    ) -> "TensorSpec":
        """Creates a new 2D tensor specification (commonly used for matrices)."""
        return cls(dimensions=[rows, cols], layout=layout, data_type=data_type)

    def get_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the tensor as a tuple."""
        return tuple(self.dimensions)

    def get_size(self) -> int:
        """Returns the total number of elements in the tensor."""
        if not self.dimensions:
            return 0
        size = 1
        for dim in self.dimensions:
            size *= dim
        return size

    def create_polars_frame(self, data=None) -> Optional["pl.DataFrame"]:
        """
        Create a Polars DataFrame with the specified shape and data type.

        Args:
            data: Optional data to initialize the DataFrame with

        Returns:
            A Polars DataFrame or None if Polars is not available
        """

        # For 2D tensors (matrices), create a DataFrame directly
        if len(self.dimensions) == 2:
            rows, cols = self.dimensions
            dtype = DataType.get_polars_dtype(self.data_type)

            if data is not None:
                # Convert data to DataFrame if provided
                if isinstance(data, list):
                    return pl.DataFrame(data)
                else:
                    # Initialize with zeros if data type is not compatible
                    return pl.DataFrame(
                        {f"col_{i}": [0.0] * rows for i in range(cols)}
                    ).cast({f"col_{i}": dtype for i in range(cols)})
            else:
                # Initialize with zeros
                return pl.DataFrame(
                    {f"col_{i}": [0.0] * rows for i in range(cols)}
                ).cast({f"col_{i}": dtype for i in range(cols)})
        else:
            # For non-2D tensors, we need more complex handling
            # This is a simplification; for real use, more sophisticated handling would be needed
            warnings.warn(
                "Non-2D tensors are not fully supported for Polars DataFrame creation."
            )
            return None


@dataclass
class GemmParams:
    """
    Parameters specific to GEMM (General Matrix Multiplication) operations.

    Represents the operation: C = alpha * A * B + beta * C
    where A, B, and C are matrices with the specified layouts.

    Attributes:
        m: Number of rows in matrices A and C
        n: Number of columns in matrices B and C
        k: Number of columns in matrix A / rows in matrix B
        alpha: Scalar multiplier for A*B
        beta: Scalar multiplier for C
        layout_a: Memory layout for matrix A
        layout_b: Memory layout for matrix B
        layout_c: Memory layout for matrix C
    """

    m: int
    n: int
    k: int
    alpha: float = 1.0
    beta: float = 0.0
    layout_a: Layout = Layout.ROW_MAJOR
    layout_b: Layout = Layout.ROW_MAJOR
    layout_c: Layout = Layout.ROW_MAJOR

    def validate(self) -> bool:
        """Validate that the GEMM parameters are consistent."""
        return self.m > 0 and self.n > 0 and self.k > 0


class OperationParams:
    """
    Base class for operation parameters.
    All specific operation parameter classes should inherit from this.
    """

    pass


class GemmOperationParams(OperationParams):
    """Wrapper class for GEMM operation parameters."""

    def __init__(self, params: GemmParams):
        self.params = params


class OperationInputs:
    """Base class for operation inputs."""

    pass


class GemmInputs(OperationInputs):
    """Input tensors for GEMM operations."""

    def __init__(self, a: TensorSpec, b: TensorSpec):
        self.a = a
        self.b = b


class OperationOutputs:
    """Base class for operation outputs."""

    pass


class GemmOutputs(OperationOutputs):
    """Output tensors for GEMM operations."""

    def __init__(self, c: TensorSpec):
        self.c = c


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
