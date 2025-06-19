"""
Core data types for kernel profiling.

This module defines the fundamental data types and structures used
throughout the kernel_scan library for GPU kernel profiling.
"""

from enum import Enum, auto

###################################################################################
#                                                                                 #
#      IN ORDER TO AVOID CIRCULAR IMPORTS                                         #
#      THIS FILE SHOULD NEVER IMPORT ANYTHING FROM THE KERNEL_SCAN LIBRARY        #
#                                                                                 #
###################################################################################


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


class OperationOutputs:
    """Base class for operation outputs."""

    pass


class OperationInputs:
    """Base class for operation inputs."""

    pass


class OperationParams:
    """Base class for operation parameters."""

    pass
