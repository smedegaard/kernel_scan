"""
Kernel specification implementation.

This module provides the KernelSpec class and related functionality for
specifying GPU kernel operations to be profiled.
"""

import json
import subprocess
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import polars as pl

from kernel_scan.core.errors import (
    MissingOperationTypeError,
    UnsupportedOperationTypeError,
)
from kernel_scan.core.logging import get_logger
from kernel_scan.core.types import (
    DataType,
    Layout,
    OperationInputs,
    OperationOutputs,
    OperationParams,
    OperationType,
)

log = get_logger(__name__)

_OPERATION_BUILDERS: Dict[OperationType, type["KernelSpecBuilder"]] = {}


def register_operation_builder(
    operation_type: OperationType, builder_class: type["KernelSpecBuilder"]
):
    """
    Register a builder class for a specific operation type.

    This function should be called by API modules to register their builders.
    """
    _OPERATION_BUILDERS[operation_type] = builder_class


class KernelSpecBuilder(ABC):
    """
    Abstract builder pattern implementation for creating KernelSpec objects.

    This class provides a fluent interface for constructing KernelSpec objects
    with a chain of method calls. Concrete implementations handle operation-specific logic.
    """

    def __init__(self):
        self._operation_type: Optional[OperationType] = None
        self._data_type: Optional[DataType] = None
        self._operation_params: Optional[OperationParams] = None
        self._inputs: Dict[str, TensorSpec] = {}
        self._outputs: Dict[str, TensorSpec] = {}
        self._iterations: int = 100
        self._name: Optional[str] = None
        self._workspace_size: Optional[int] = None

    def operation_type(self, op_type: OperationType) -> "KernelSpecBuilder":
        """Set the operation type."""
        self._operation_type = op_type
        return self

    def data_type(self, dtype: DataType) -> "KernelSpecBuilder":
        """Set the data type."""
        self._data_type = dtype
        return self

    def operation_params(self, params: OperationParams) -> "KernelSpecBuilder":
        """Set operation-specific parameters."""
        self._operation_params = params
        return self

    def inputs(self, **kwargs: "TensorSpec") -> "KernelSpecBuilder":
        """Set input tensor specifications."""
        self._inputs.update(kwargs)
        return self

    def outputs(self, **kwargs: "TensorSpec") -> "KernelSpecBuilder":
        """Set output tensor specifications."""
        self._outputs.update(kwargs)
        return self

    def iterations(self, iterations: int) -> "KernelSpecBuilder":
        """Set the number of profiling iterations."""
        self._iterations = iterations
        return self

    def name(self, name: str) -> "KernelSpecBuilder":
        """Set an optional name for the kernel specification."""
        self._name = name
        return self

    def workspace_size(self, size: int) -> "KernelSpecBuilder":
        """Set an optional workspace size in bytes."""
        self._workspace_size = size
        return self

    def get_operation_type(self) -> Optional[OperationType]:
        """Get the operation type."""
        return self._operation_type

    def get_data_type(self) -> Optional[DataType]:
        """Get the data type."""
        return self._data_type

    def get_operation_params(self) -> Optional[OperationParams]:
        """Get operation-specific parameters."""
        return self._operation_params

    def get_inputs(self) -> Dict[str, "TensorSpec"]:
        """Get input tensor specifications."""
        return self._inputs

    def get_outputs(self) -> Dict[str, "TensorSpec"]:
        """Get output tensor specifications."""
        return self._outputs

    def get_iterations(self) -> int:
        """Get the number of profiling iterations."""
        return self._iterations

    def get_name(self) -> Optional[str]:
        """Get the name for the kernel specification."""
        return self._name

    def get_workspace_size(self) -> Optional[int]:
        """Get the workspace size in bytes."""
        return self._workspace_size

    @abstractmethod
    def _validate_config(self):
        """
        Validate that the builder configuration is complete and consistent.

        This method should be implemented by concrete builder classes to perform
        operation-specific validation of the builder's state before building.

        Raises:
            Various KernelSpecError subclasses if validation fails
        """
        pass

    @abstractmethod
    def build(self) -> "KernelSpec":
        """
        Build and return a KernelSpec object.

        This method should call _validate_config() first to ensure the builder
        state is valid before constructing the KernelSpec.

        Returns:
            A fully constructed KernelSpec object

        Raises:
            Various KernelSpecError subclasses if required parameters are missing
        """
        pass


class GenericKernelSpecBuilder(KernelSpecBuilder):
    """
    Generic builder that switches to operation-specific builders when operation_type() is called.
    """

    def __init__(self):
        super().__init__()
        self._specific_builder: Optional[KernelSpecBuilder] = None

    def operation_type(self, op_type: OperationType) -> KernelSpecBuilder:
        """Switch to operation-specific builder."""
        if op_type not in _OPERATION_BUILDERS:
            available_types = list(_OPERATION_BUILDERS.keys())
            raise UnsupportedOperationTypeError(
                f"No builder registered for operation type: {op_type}. "
                f"Available types: {available_types}."
            )

        # Create specific builder and transfer state
        builder_class = _OPERATION_BUILDERS[op_type]
        self._specific_builder = builder_class()

        # Transfer state
        self._specific_builder._data_type = self._data_type
        self._specific_builder._operation_params = self._operation_params
        self._specific_builder._inputs = self._inputs.copy()
        self._specific_builder._outputs = self._outputs.copy()
        self._specific_builder._iterations = self._iterations
        self._specific_builder._name = self._name
        self._specific_builder._workspace_size = self._workspace_size
        self._specific_builder._operation_type = op_type

        return self._specific_builder

    def _validate_config(self):
        """Validate that operation type has been set."""
        raise MissingOperationTypeError(
            "Operation type not set. Call operation_type() to specify the operation."
        )

    def build(self) -> "KernelSpec":
        """Build - should not be called on generic builder."""
        self._validate_config()


@dataclass
class KernelSpec(ABC):
    """
    Abstract base class for GPU kernel specifications to be profiled.

    This class describes all aspects of a kernel operation including:
    - Data type (FP32, FP16, etc.)
    - Input/output tensor specifications
    - Profiling configuration

    Concrete implementations should define specific operation types
    (GEMM, Convolution, etc.) and their associated parameters.

    Attributes:
        data_type: Data type for the operation
        iterations: Number of iterations to run for profiling
        name: Optional name for the kernel specification
        workspace_size: Optional workspace size in bytes
    """

    data_type: DataType
    iterations: int = 10
    name: Optional[str] = None
    workspace_size: Optional[int] = None

    @property
    @abstractmethod
    def operation_type(self) -> OperationType:
        """Return the operation type for this kernel specification."""
        pass

    @property
    @abstractmethod
    def operation_params(self) -> OperationParams:
        """Return the operation-specific parameters."""
        pass

    @property
    @abstractmethod
    def inputs(self) -> OperationInputs:
        """Return the input tensor specifications."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> OperationOutputs:
        """Return the output tensor specifications."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that the kernel specification is consistent.

        Returns:
            True if the specification is valid

        Raises:
            Various KernelSpecError subclasses if validation fails
        """
        pass

    @classmethod
    def builder(
        cls, operation_type: Optional[OperationType] = None
    ) -> KernelSpecBuilder:
        """
        Create a new KernelSpecBuilder instance.

        Args:
            operation_type: Optional operation type. If None, returns generic builder.
        """
        if operation_type is None:
            return GenericKernelSpecBuilder()

        if operation_type not in _OPERATION_BUILDERS:
            available_types = list(_OPERATION_BUILDERS.keys())
            raise UnsupportedOperationTypeError(
                f"No builder registered for operation type: {operation_type}. "
                f"Available types: {available_types}."
            )

        builder_class = _OPERATION_BUILDERS[operation_type]
        return builder_class()


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

    def as_dataframe(self, data=None) -> Optional["pl.DataFrame"]:
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
class AcceleratorSpec:
    """
    Accelerator hardware specifications for performance analysis.

    Attributes:
        name: Name or model of the accelerator
        peak_memory_bandwidth_gbps: Peak memory bandwidth in GB/s
        peak_performance: Dict mapping data types to TFLOPS values
        memory_size_gb: Size of accelerator memory in GB
        additional_specs: Additional hardware specifications
    """

    name: str = "Unknown"
    peak_memory_bandwidth_gbps: Optional[float] = None
    peak_performance: Dict[DataType, float] = field(default_factory=dict)
    memory_size_gb: Optional[float] = None
    additional_specs: Dict[str, any] = field(default_factory=dict)

    @property
    def peak_bandwidth(self) -> float:
        """Return peak memory bandwidth in GB/s."""
        return (
            self.peak_memory_bandwidth_gbps
            if self.peak_memory_bandwidth_gbps is not None
            else 0.0
        )

    def get_peak_compute(self, precision: DataType = DataType.FLOAT32) -> float:
        """
        Get peak compute performance for the specified precision.

        Args:
            precision: Precision format to query

        Returns:
            Peak compute performance in TFLOPS
        """

        if precision in self.peak_performance:
            return self.peak_performance[precision]
        else:
            log.warning(
                f"No peak performance data for {precision.name} on {self.name}. Using FLOAT32 as fallback."
            )
            return self.peak_performance.get(DataType.FLOAT32, 0.0)

    @staticmethod
    def detect_hardware() -> "AcceleratorSpec":
        """
        Detect GPU hardware information and create an AcceleratorSpec instance.

        This method attempts to identify the GPU hardware by querying system tools
        and creates an AcceleratorSpec instance with the detected values.

        Returns:
            AcceleratorSpec instance with detected hardware information
        """
        # Start with unknown values
        specs = AcceleratorSpec()

        # Try to detect the GPU from system
        detected_vendor = None
        detected_model = None

        # First try AMD GPUs with rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--showallinfo", "--json"],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                try:
                    rocm_info = json.loads(result.stdout)
                    if isinstance(rocm_info, dict):
                        for card_id, value in rocm_info.items():
                            # Find the primary GPU - we'll use the first discrete GPU
                            device_name = rocm_info[card_id]["Device Name"]
                            # Check if this is a known AMD GPU
                            detected_vendor = "AMD"
                            log.info(f"Detected AMD GPU: {device_name}")

                            # Look for exact match
                            if device_name in GPU_SPECS["AMD"]:
                                detected_model = device_name
                                break

                            # If we found a match, no need to check other cards
                            if detected_model:
                                break
                except (json.JSONDecodeError, KeyError) as e:
                    log.info(f"Failed to parse rocm-smi output: {e}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.info(f"Failed to run rocm-smi: {e}")

        # todo: If AMD detection failed, try NVIDIA GPUs

        # If we detected a known GPU, update the AcceleratorSpec
        if detected_vendor and detected_model:
            log.info(
                f"Found matching GPU model: {detected_model} from vendor {detected_vendor}"
            )
            gpu_specs = GPU_SPECS[detected_vendor][detected_model]

            # Create AcceleratorSpec with detected values
            specs = AcceleratorSpec(
                name=gpu_specs["name"],
                peak_memory_bandwidth_gbps=gpu_specs["peak_memory_bandwidth_gbps"],
                peak_performance=gpu_specs["peak_performance"].copy(),
                memory_size_gb=gpu_specs["memory_size_gb"],
                additional_specs=gpu_specs["additional_specs"].copy(),
            )

            # Add vendor information
            specs.additional_specs["vendor"] = detected_vendor
        else:
            # If we couldn't find a match in our database, log a warning
            if detected_vendor:
                log.warning(
                    f"Detected {detected_vendor} GPU, but couldn't match it to a known model in the database."
                )
                specs.additional_specs["vendor"] = detected_vendor
            else:
                log.warning("Could not detect GPU vendor or model.")

        return specs


# GPU specifications map organized by vendor and model
GPU_SPECS = {
    "AMD": {
        "Radeon RX 7900 XTX": {
            "name": "Radeon RX 7900 XTX",
            "peak_memory_bandwidth_gbps": 960.0,
            "peak_performance": {
                DataType.FLOAT32: 61.4,  # FP32 TFLOPS
                DataType.FLOAT16: 122.8,  # FP16 TFLOPS (2x FP32)
                DataType.INT8: 245.6,  # INT8 TOPS (4x FP32)
            },
            "memory_size_gb": 24.0,
            "additional_specs": {
                "stream_processors": 12288,
                "compute_units": 96,
                "memory_type": "GDDR6",
                "memory_bus_width": 384,
                "architecture": "RDNA 3",
                "release_year": 2022,
            },
        },
        "Instinct MI250X": {
            "name": "AMD Instinct MI250X",
            "peak_memory_bandwidth_gbps": 3200.0,
            "peak_performance": {
                DataType.FLOAT64: 47.9,  # FP64 TFLOPS
                DataType.FLOAT32: 95.7,  # FP32 TFLOPS
                DataType.FLOAT16: 383.0,  # FP16 TFLOPS
                DataType.BFLOAT16: 383.0,  # BF16 TFLOPS
                DataType.INT8: 766.0,  # INT8 TOPS
            },
            "memory_size_gb": 128.0,
            "additional_specs": {
                "stream_processors": 14080,
                "compute_units": 220,
                "memory_type": "HBM2e",
                "memory_bus_width": 8192,
                "architecture": "CDNA 2",
                "release_year": 2021,
            },
        },
        "Instinct MI300X": {
            "name": "AMD Instinct MI300X",
            "peak_memory_bandwidth_gbps": 5300.0,  # 5.3 TB/s
            "peak_performance": {
                DataType.FLOAT64: 152.8,  # FP64 TFLOPS
                DataType.FLOAT32: 305.6,  # FP32 TFLOPS
                DataType.FLOAT16: 611.2,  # FP16 TFLOPS
                DataType.BFLOAT16: 611.2,  # BF16 TFLOPS
                DataType.INT8: 1222.4,  # INT8 TOPS
            },
            "memory_size_gb": 192.0,
            "additional_specs": {
                "stream_processors": 304 * 256,  # 304 CUs * 256 SPs per CU
                "compute_units": 304,
                "memory_type": "HBM3",
                "memory_bus_width": 8192,
                "architecture": "CDNA 3",
                "release_year": 2023,
                "chiplets": 8,  # 8 GCD chiplets
                "matrix_engines": 304,  # One per CU
            },
        },
    },
    "NVIDIA": {
        "A100": {
            "name": "NVIDIA A100",
            "peak_memory_bandwidth_gbps": 2039.0,
            "peak_performance": {
                DataType.FLOAT64: 9.7,  # FP64 TFLOPS
                DataType.FLOAT32: 19.5,  # FP32 TFLOPS
                DataType.FLOAT16: 312.0,  # FP16 TFLOPS (with Tensor Cores)
                DataType.BFLOAT16: 312.0,  # BF16 TFLOPS (with Tensor Cores)
                DataType.INT8: 624.0,  # INT8 TOPS (with Tensor Cores)
                DataType.INT4: 1248.0,  # INT4 TOPS (with Tensor Cores)
            },
            "memory_size_gb": 80.0,
            "additional_specs": {
                "cuda_cores": 6912,
                "tensor_cores": 432,
                "memory_type": "HBM2e",
                "memory_bus_width": 5120,
                "architecture": "Ampere",
                "release_year": 2020,
            },
        },
        "H100": {
            "name": "NVIDIA H100",
            "peak_memory_bandwidth_gbps": 3350.0,
            "peak_performance": {
                DataType.FLOAT64: 67.0,  # FP64 TFLOPS (with Tensor Cores)
                DataType.FLOAT32: 67.0,  # FP32 TFLOPS
                DataType.FLOAT16: 989.0,  # FP16 TFLOPS (with Tensor Cores)
                DataType.BFLOAT16: 989.0,  # BF16 TFLOPS (with Tensor Cores)
                DataType.INT8: 1979.0,  # INT8 TOPS (with Tensor Cores)
            },
            "memory_size_gb": 80.0,
            "additional_specs": {
                "cuda_cores": 16896,
                "tensor_cores": 528,
                "memory_type": "HBM3",
                "memory_bus_width": 5120,
                "architecture": "Hopper",
                "release_year": 2022,
            },
        },
    },
}
