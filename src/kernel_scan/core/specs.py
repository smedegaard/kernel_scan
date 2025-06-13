"""
Kernel specification implementation.

This module provides the KernelSpec class and related functionality for
specifying GPU kernel operations to be profiled.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, cast

from kernel_scan.core.types import (
    DataType,
    GemmInputs,
    GemmOperationParams,
    GemmOutputs,
    GemmParams,
    MissingDataTypeError,
    MissingInputError,
    MissingOperationParamsError,
    MissingOperationTypeError,
    MissingOutputError,
    OperationInputs,
    OperationOutputs,
    OperationParams,
    OperationType,
    TensorSpec,
    UnsupportedOperationTypeError,
)
from kernel_scan.ops.gemm import validate_gemm_operation


@dataclass
class KernelSpec:
    """
    Specification for a GPU kernel to be profiled.

    This class describes all aspects of a kernel operation including:
    - Operation type (GEMM, etc.)
    - Data type (FP32, FP16, etc.)
    - Operation-specific parameters
    - Input/output tensor specifications
    - Profiling configuration

    Attributes:
        operation_type: Type of operation (GEMM, etc.)
        data_type: Data type for the operation
        operation_params: Parameters specific to the operation type
        inputs: Input tensor specifications
        outputs: Output tensor specifications
        iterations: Number of iterations to run for profiling
        name: Optional name for the kernel specification
        workspace_size: Optional workspace size in bytes
        extra_params: Optional dictionary of extra parameters
    """

    operation_type: OperationType
    data_type: DataType
    operation_params: OperationParams
    inputs: OperationInputs
    outputs: OperationOutputs
    iterations: int = 100
    name: Optional[str] = None
    workspace_size: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Validate that the kernel specification is consistent.

        Returns:
            True if the specification is valid

        Raises:
            Various KernelSpecError subclasses if validation fails
        """
        if self.operation_type == OperationType.GEMM:
            # Validate GEMM operation
            gemm_params = cast(GemmOperationParams, self.operation_params).params
            gemm_inputs = cast(GemmInputs, self.inputs)
            gemm_outputs = cast(GemmOutputs, self.outputs)
            return validate_gemm_operation(gemm_params, gemm_inputs, gemm_outputs)
        else:
            raise UnsupportedOperationTypeError(self.operation_type)

    @classmethod
    def builder(cls) -> "KernelSpecBuilder":
        """Create a new KernelSpecBuilder instance."""
        return KernelSpecBuilder()


class KernelSpecBuilder:
    """
    Builder pattern implementation for creating KernelSpec objects.

    This class provides a fluent interface for constructing KernelSpec objects
    with a chain of method calls.
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
        self._extra_params: Dict[str, Any] = {}

    def operation_type(self, op_type: OperationType) -> "KernelSpecBuilder":
        """Set the operation type."""
        self._operation_type = op_type
        return self

    def data_type(self, dtype: DataType) -> "KernelSpecBuilder":
        """Set the data type."""
        self._data_type = dtype
        return self

    def operation_params(self, params: Union[GemmParams]) -> "KernelSpecBuilder":
        """
        Set operation-specific parameters.

        Args:
            params: Operation parameters (e.g., GemmParams)
        """
        if isinstance(params, GemmParams):
            self._operation_params = GemmOperationParams(params)
        else:
            raise ValueError(f"Unsupported operation params type: {type(params)}")
        return self

    def inputs(self, **kwargs: TensorSpec) -> "KernelSpecBuilder":
        """
        Set input tensor specifications.

        Args:
            **kwargs: Named tensor specifications (e.g., a=tensor_spec_a, b=tensor_spec_b)
        """
        self._inputs.update(kwargs)
        return self

    def outputs(self, **kwargs: TensorSpec) -> "KernelSpecBuilder":
        """
        Set output tensor specifications.

        Args:
            **kwargs: Named tensor specifications (e.g., c=tensor_spec_c)
        """
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

    def extra_params(self, **kwargs: Any) -> "KernelSpecBuilder":
        """
        Set extra parameters.

        Args:
            **kwargs: Named extra parameters
        """
        self._extra_params.update(kwargs)
        return self

    def build(self) -> KernelSpec:
        """
        Build and return a KernelSpec object.

        Returns:
            A fully constructed KernelSpec object

        Raises:
            Various KernelSpecError subclasses if required parameters are missing
        """
        # Validate required parameters
        if self._operation_type is None:
            raise MissingOperationTypeError()

        if self._data_type is None:
            raise MissingDataTypeError()

        if self._operation_params is None:
            raise MissingOperationParamsError()

        # Create operation-specific inputs and outputs
        if self._operation_type == OperationType.GEMM:
            # Validate and create GEMM inputs
            if "a" not in self._inputs:
                raise MissingInputError("a")
            if "b" not in self._inputs:
                raise MissingInputError("b")
            if "c" not in self._outputs:
                raise MissingOutputError("c")

            inputs = GemmInputs(a=self._inputs["a"], b=self._inputs["b"])
            outputs = GemmOutputs(c=self._outputs["c"])
        else:
            raise UnsupportedOperationTypeError(self._operation_type)

        # Create and return the KernelSpec
        spec = KernelSpec(
            operation_type=self._operation_type,
            data_type=self._data_type,
            operation_params=self._operation_params,
            inputs=inputs,
            outputs=outputs,
            iterations=self._iterations,
            name=self._name,
            workspace_size=self._workspace_size,
            extra_params=self._extra_params,
        )

        # Validate the specification
        spec.validate()

        return spec
