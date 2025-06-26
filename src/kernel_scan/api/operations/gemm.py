"""
GEMM operations implementation.

This module implements the General Matrix Multiplication (GEMM) operations
for the kernel_scan library. It provides the necessary functionality to
define and validate GEMM operations for GPU profiling.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
)

import polars as pl

from kernel_scan.api.profiler import Profiler
from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.engine import EngineType
from kernel_scan.core.errors import (
    IncompatibleDataTypesError,
    IncompatibleDimensionsError,
    IncompatibleLayoutError,
    InvalidTensorShapeError,
    MissingDataTypeError,
    MissingInputError,
    MissingOperationParamsError,
    MissingOutputError,
    OperationParameterMismatchError,
    UnsupportedOperationTypeError,
)
from kernel_scan.core.logging import get_logger
from kernel_scan.core.plots import OperationPlotter
from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.specs import (
    KernelSpec,
    KernelSpecBuilder,
    TensorSpec,
    register_operation_builder,
)
from kernel_scan.core.types import (
    DataType,
    Layout,
    OperationInputs,
    OperationOutputs,
    OperationParams,
    OperationType,
)
from kernel_scan.core.units import Byte, Flops, FlopsPerByte, GigaBytesPerSecond

log = get_logger("gemm")


@dataclass
class GemmParams(OperationParams):
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


class GemmOperationParams(OperationParams):
    """Wrapper class for GEMM operation parameters."""

    def __init__(self, params: GemmParams):
        self.params = params


class GemmInputs(OperationInputs):
    """Input tensors for GEMM operations."""

    def __init__(self, a: TensorSpec, b: TensorSpec):
        self.a = a
        self.b = b


class GemmOutputs(OperationOutputs):
    """Output tensors for GEMM operations."""

    def __init__(self, c: TensorSpec):
        self.c = c


@dataclass
class GemmKernelSpec(KernelSpec):
    """
    Kernel specification for GEMM (General Matrix Multiplication) operations.

    Attributes:
        gemm_params: GEMM-specific operation parameters
        gemm_inputs: GEMM input tensors (matrices A and B)
        gemm_outputs: GEMM output tensor (matrix C)
    """

    # Use Optional with None defaults to fix dataclass field ordering
    gemm_params: Optional[GemmOperationParams] = None
    gemm_inputs: Optional[GemmInputs] = None
    gemm_outputs: Optional[GemmOutputs] = None

    def __post_init__(self):
        """Validate that required fields are provided."""
        if self.gemm_params is None:
            raise ValueError("gemm_params is required for GemmKernelSpec")
        if self.gemm_inputs is None:
            raise ValueError("gemm_inputs is required for GemmKernelSpec")
        if self.gemm_outputs is None:
            raise ValueError("gemm_outputs is required for GemmKernelSpec")

    @property
    def operation_type(self) -> OperationType:
        """Return the operation type for GEMM."""
        return OperationType.GEMM

    @property
    def operation_params(self) -> OperationParams:
        """Return the GEMM operation parameters."""
        return self.gemm_params.params

    @operation_params.setter
    def operation_params(self, value: OperationParams):
        """Set the GEMM operation parameters."""
        self.gemm_params = value

    @property
    def inputs(self) -> OperationInputs:
        """Return the GEMM input specifications."""
        return self.gemm_inputs

    @property
    def outputs(self) -> OperationOutputs:
        """Return the GEMM output specifications."""
        return self.gemm_outputs

    def validate(self) -> bool:
        """
        Validate that the GEMM kernel specification is consistent.

        Returns:
            True if the specification is valid

        Raises:
            Various KernelSpecError subclasses if validation fails
        """
        # Extract parameters
        params = self.gemm_params.params
        a_spec = self.gemm_inputs.a
        b_spec = self.gemm_inputs.b
        c_spec = self.gemm_outputs.c

        # Validate data types consistency
        if a_spec.data_type != self.data_type:
            raise IncompatibleDataTypesError(
                f"Input A data type {a_spec.data_type} doesn't match kernel data type {self.data_type}"
            )
        if b_spec.data_type != self.data_type:
            raise IncompatibleDataTypesError(
                f"Input B data type {b_spec.data_type} doesn't match kernel data type {self.data_type}"
            )
        if c_spec.data_type != self.data_type:
            raise IncompatibleDataTypesError(
                f"Output C data type {c_spec.data_type} doesn't match kernel data type {self.data_type}"
            )

        # Validate tensor dimensions
        if len(a_spec.dimensions) != 2:
            raise InvalidTensorShapeError(
                f"Matrix A must be 2D, got {len(a_spec.dimensions)}D"
            )
        if len(b_spec.dimensions) != 2:
            raise InvalidTensorShapeError(
                f"Matrix B must be 2D, got {len(b_spec.dimensions)}D"
            )
        if len(c_spec.dimensions) != 2:
            raise InvalidTensorShapeError(
                f"Matrix C must be 2D, got {len(c_spec.dimensions)}D"
            )

        # Get actual dimensions
        a_rows, a_cols = a_spec.dimensions
        b_rows, b_cols = b_spec.dimensions
        c_rows, c_cols = c_spec.dimensions

        # Validate dimension compatibility
        if a_cols != b_rows:
            raise IncompatibleDimensionsError(
                f"Matrix multiplication dimension mismatch: A columns ({a_cols}) != B rows ({b_rows})"
            )
        if a_rows != c_rows:
            raise IncompatibleDimensionsError(
                f"Output dimension mismatch: A rows ({a_rows}) != C rows ({c_rows})"
            )
        if b_cols != c_cols:
            raise IncompatibleDimensionsError(
                f"Output dimension mismatch: B columns ({b_cols}) != C columns ({c_cols})"
            )

        # Validate against operation parameters
        if params.m != c_rows:
            raise OperationParameterMismatchError(
                f"Parameter M ({params.m}) doesn't match output rows ({c_rows})"
            )
        if params.n != c_cols:
            raise OperationParameterMismatchError(
                f"Parameter N ({params.n}) doesn't match output columns ({c_cols})"
            )
        if params.k != a_cols:
            raise OperationParameterMismatchError(
                f"Parameter K ({params.k}) doesn't match inner dimension ({a_cols})"
            )

        return True


class GemmKernelSpecBuilder(KernelSpecBuilder):
    """
    GEMM-specific implementation of KernelSpecBuilder.

    This builder knows how to construct GemmKernelSpec objects with proper
    validation and type conversion for GEMM operations.
    """

    def operation_params(self, params) -> "GemmKernelSpecBuilder":
        """
        Set GEMM operation parameters.

        Args:
            params: Can be GemmParams or GemmOperationParams
        """
        if isinstance(params, GemmParams):
            self._operation_params = GemmOperationParams(params)
        elif isinstance(params, GemmOperationParams):
            self._operation_params = params
        else:
            raise ValueError(f"Unsupported GEMM operation params type: {type(params)}")
        return self

    def _validate_config(self):
        """
        Validate that the builder configuration is complete and consistent for GEMM.

        Raises:
            Various KernelSpecError subclasses if validation fails
        """
        # Validate operation type
        if self._operation_type is None:
            # Auto-set to GEMM if not specified
            self._operation_type = OperationType.GEMM
        elif self._operation_type != OperationType.GEMM:
            raise UnsupportedOperationTypeError(
                f"GemmKernelSpecBuilder only supports GEMM operations, got {self._operation_type}"
            )

        # Validate required parameters
        if self._data_type is None:
            raise MissingDataTypeError()

        if self._operation_params is None:
            raise MissingOperationParamsError()

        # Validate GEMM-specific input/output requirements
        if "a" not in self._inputs:
            raise MissingInputError("a")
        if "b" not in self._inputs:
            raise MissingInputError("b")
        if "c" not in self._outputs:
            raise MissingOutputError("c")

        # Validate that parameters match the operation type
        if not isinstance(self._operation_params, GemmOperationParams):
            raise ValueError(
                f"Expected GemmOperationParams for GEMM operation, got {type(self._operation_params)}"
            )

    def build(self) -> KernelSpec:
        """
        Build and return a GemmKernelSpec object.

        Returns:
            A fully constructed GemmKernelSpec object

        Raises:
            Various KernelSpecError subclasses if required parameters are missing
        """
        # Validate configuration first
        self._validate_config()

        # Create GEMM-specific inputs and outputs
        inputs = GemmInputs(a=self._inputs["a"], b=self._inputs["b"])
        outputs = GemmOutputs(c=self._outputs["c"])

        # Create GemmKernelSpec
        spec = GemmKernelSpec(
            data_type=self._data_type,
            iterations=self._iterations,
            name=self._name,
            workspace_size=self._workspace_size,
            gemm_params=self._operation_params,
            gemm_inputs=inputs,
            gemm_outputs=outputs,
        )

        # Final validation of the constructed spec
        spec.validate()

        return spec


def validate_gemm_operation(
    params: GemmParams, inputs: GemmInputs, outputs: GemmOutputs
) -> bool:
    """
    Validate that GEMM operation parameters, inputs, and outputs are consistent.

    Args:
        params: The GEMM operation parameters
        inputs: The input tensors (A and B)
        outputs: The output tensor (C)

    Returns:
        True if the operation is valid

    Raises:
        InvalidTensorShapeError: If any tensor shape is invalid
        IncompatibleDimensionsError: If tensor dimensions don't match the operation
        IncompatibleLayoutError: If tensor layouts don't match the operation parameters
        IncompatibleDataTypesError: If tensor data types are inconsistent
    """
    # Validate input dimensions
    if len(inputs.a.dimensions) != 2:
        raise InvalidTensorShapeError(
            f"Matrix A must be 2D, got {len(inputs.a.dimensions)}D"
        )
    if len(inputs.b.dimensions) != 2:
        raise InvalidTensorShapeError(
            f"Matrix B must be 2D, got {len(inputs.b.dimensions)}D"
        )
    if len(outputs.c.dimensions) != 2:
        raise InvalidTensorShapeError(
            f"Matrix C must be 2D, got {len(outputs.c.dimensions)}D"
        )

    # Extract dimensions based on layout
    if inputs.a.layout == Layout.ROW_MAJOR:
        a_rows, a_cols = inputs.a.dimensions
    else:
        a_cols, a_rows = inputs.a.dimensions

    if inputs.b.layout == Layout.ROW_MAJOR:
        b_rows, b_cols = inputs.b.dimensions
    else:
        b_cols, b_rows = inputs.b.dimensions

    if outputs.c.layout == Layout.ROW_MAJOR:
        c_rows, c_cols = outputs.c.dimensions
    else:
        c_cols, c_rows = outputs.c.dimensions

    # Validate dimensions match the GEMM operation
    if a_rows != params.m:
        raise IncompatibleDimensionsError(
            f"Matrix A rows ({a_rows}) doesn't match GEMM m parameter ({params.m})"
        )

    if a_cols != params.k:
        raise IncompatibleDimensionsError(
            f"Matrix A columns ({a_cols}) doesn't match GEMM k parameter ({params.k})"
        )

    if b_rows != params.k:
        raise IncompatibleDimensionsError(
            f"Matrix B rows ({b_rows}) doesn't match GEMM k parameter ({params.k})"
        )

    if b_cols != params.n:
        raise IncompatibleDimensionsError(
            f"Matrix B columns ({b_cols}) doesn't match GEMM n parameter ({params.n})"
        )

    if c_rows != params.m:
        raise IncompatibleDimensionsError(
            f"Matrix C rows ({c_rows}) doesn't match GEMM m parameter ({params.m})"
        )

    if c_cols != params.n:
        raise IncompatibleDimensionsError(
            f"Matrix C columns ({c_cols}) doesn't match GEMM n parameter ({params.n})"
        )

    # Validate layouts
    if inputs.a.layout != params.layout_a:
        raise IncompatibleLayoutError(
            f"Matrix A layout ({inputs.a.layout}) doesn't match specified layout ({params.layout_a})"
        )

    if inputs.b.layout != params.layout_b:
        raise IncompatibleLayoutError(
            f"Matrix B layout ({inputs.b.layout}) doesn't match specified layout ({params.layout_b})"
        )

    if outputs.c.layout != params.layout_c:
        raise IncompatibleLayoutError(
            f"Matrix C layout ({outputs.c.layout}) doesn't match specified layout ({params.layout_c})"
        )

    # Validate data types consistency
    if inputs.a.data_type != inputs.b.data_type:
        raise IncompatibleDataTypesError(
            f"Matrix A data type ({inputs.a.data_type}) doesn't match Matrix B data type ({inputs.b.data_type})"
        )

    if inputs.a.data_type != outputs.c.data_type:
        raise IncompatibleDataTypesError(
            f"Input matrices data type ({inputs.a.data_type}) doesn't match output matrix data type ({outputs.c.data_type})"
        )

    return True


def calculate_flops(params: GemmParams) -> Flops:
    """
    Calculate the number of floating-point operations for a GEMM operation.

    For matrix multiplication C = alpha * A * B + beta * C:
    - Each element in C requires k multiply-adds (2 operations per multiply-add)
    - If alpha != 1.0, add one multiplication per element in C
    - If beta != 0.0, add one multiplication per element in C

    Args:
        params: The GEMM operation parameters

    Returns:
        The number of floating-point operations
    """
    # Core GEMM operations: 2 operations per multiply-add for each element in output matrix
    flops = 2 * params.m * params.n * params.k

    # Account for alpha scaling if alpha is not 1.0
    if params.alpha != 1.0:
        flops += params.m * params.n

    # Account for beta scaling if beta is not 0.0
    if params.beta != 0.0:
        flops += params.m * params.n

    return Flops(flops)


def calculate_bytes_moved(params: GemmParams, dtype_size: int) -> Byte:
    """
    Calculate bytes moved for GEMM: C = alpha * A * B + beta * C.

    Memory access rules:
    - Always read A (m x k) and B (k x n)
    - Always write C (m x n)
    - Read C only if beta != 0 (for beta * C)

    Args:
        params: GEMM parameters (m, n, k, alpha, beta)
        dtype_size: Size of data type in bytes

    Returns:
        Total bytes moved
    """
    a_bytes = params.m * params.k * dtype_size  # Read A
    b_bytes = params.k * params.n * dtype_size  # Read B
    c_bytes = params.m * params.n * dtype_size  # Always write C

    # Read C only if beta != 0
    c_read_bytes = c_bytes if params.beta != 0.0 else 0

    return Byte(a_bytes + b_bytes + c_read_bytes + c_bytes)


def calculate_arithmetic_intensity(params: GemmParams, dtype: DataType) -> FlopsPerByte:
    """
    Calculate the arithmetic intensity (FLOPs per byte) for a GEMM operation.

    Arithmetic intensity is defined as the ratio of the number of floating-point
    operations to the number of bytes moved.

    Args:
        params: The GEMM operation parameters
        dtype: The data type used in the operation

    Returns:
        The arithmetic intensity in FLOPs/byte
    """
    flops = calculate_flops(params)
    dtype_size = DataType.get_size_bytes(dtype)
    bytes_moved = calculate_bytes_moved(params, dtype_size)

    flops_per_byte_value = flops.value / bytes_moved.value
    return FlopsPerByte(flops_per_byte_value)


def verify_gemm_result(
    a: pl.DataFrame,
    b: pl.DataFrame,
    c: pl.DataFrame,
    c_result: pl.DataFrame,
    params: GemmParams,
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify that a GEMM operation result is correct using Polars.

    Args:
        a: Input matrix A (Polars DataFrame)
        b: Input matrix B (Polars DataFrame)
        c: Input matrix C before the operation (Polars DataFrame)
        c_result: Result matrix C after the operation (Polars DataFrame)
        params: The GEMM operation parameters
        tolerance: Tolerance for floating point comparison

    Returns:
        True if the result is correct within the tolerance
    """
    raise NotImplementedError("GEMM verification not implemented yet")


class GemmTestCase(NamedTuple):
    """A single GEMM test configuration."""

    data_type: DataType
    m: int
    n: int
    k: int

    @property
    def name(self) -> str:
        """Generate a descriptive name for this test case."""
        return f"GEMM_M{self.m}_N{self.n}_K{self.k}_{self.data_type.name}"

    @property
    def output_path(self) -> Path:
        """Generate the output path relative to a base directory."""
        return Path(
            f"{self.data_type.name}/NK{self.n}/gemm_M{self.m}_N{self.n}_K{self.k}_{self.data_type.name.lower()}.jsonl"
        )


class GemmScan:
    """
    A fluent API for scanning GEMM performance across multiple dimensions and data types.

    Example usage:
        scanner = GemmScan()
        results = (scanner
            .with_data_types([DataType.FLOAT16, DataType.FLOAT32])
            .for_n_values([1024, 2048, 4096])
            .for_m_values([1, 2, 4, 8, 16])
            .with_k_equals_n()
            .with_engine_type(EngineType.COMPOSABLE_KERNEL)  # Mandatory
            .iterations(10)
            .warmup(5)
            .run())
    """

    def __init__(self, profile_config: Optional[ProfileConfig] = None):
        """
        Initialize the GEMM scanner with default configuration.

        Args:
            profile_config: Optional profile configuration. If not provided, a default one will be used.
        """
        # Core scan parameters
        self.engine_type: Optional[EngineType] = None
        self.data_types: List[DataType] = []
        self.m_values: List[int] = []
        self.n_values: List[int] = []
        self.k_values: List[int] = []
        self.nk_linked: bool = False
        self._iterations: int = 100
        self.warmup_iterations: int = 10
        self.layout_a: Layout = Layout.ROW_MAJOR
        self.layout_b: Layout = Layout.ROW_MAJOR
        self.layout_c: Layout = Layout.ROW_MAJOR
        self.alpha: float = 1.0
        self.beta: float = 0.0
        self.output_dir: Union[str, Path] = Path("results")

        # Advanced configuration
        self.growing_dimension: Optional[str] = None
        self.dimension_relationships: Dict[str, Callable[[int], int]] = {}

        # Internal state
        self._results: Dict[str, List[Any]] = {}
        self._base_output_dir: Optional[Path] = None
        self._plots_dir: Optional[Path] = None

        # Profile configuration
        self.profile_config = profile_config or ProfileConfig.create_default()

    def with_engine_type(self, engine_type: EngineType) -> "GemmScan":
        """
        Set the engine type to use for profiling.
        This builder method is mandatory before calling run().
        """
        self.engine_type = engine_type
        return self

    def with_data_types(self, data_types: List[DataType]) -> "GemmScan":
        """Set the data types to scan."""
        self.data_types = data_types
        return self

    def for_n_values(self, n_values: List[int]) -> "GemmScan":
        """Set the N dimension values to scan."""
        self.n_values = n_values
        return self

    def for_m_values(self, m_values: List[int]) -> "GemmScan":
        """Set the M dimension values to scan."""
        self.m_values = m_values
        return self

    def for_k_values(self, k_values: List[int]) -> "GemmScan":
        """Set the K dimension values to scan."""
        self.k_values = k_values
        return self

    def with_k_equals_n(self) -> "GemmScan":
        """Configure the scan to use K = N for all test cases."""
        self.nk_linked = True
        return self

    def iterations(self, count: int) -> "GemmScan":
        """Set the number of profiling iterations."""
        self._iterations = count
        # Update the profile_config with the same number of iterations
        self.profile_config.iterations = count
        return self

    def warmup(self, count: int) -> "GemmScan":
        """Set the number of warmup iterations."""
        self.warmup_iterations = count
        # Update the profile_config with the same number of warmup iterations
        self.profile_config.warmup_iterations = count
        return self

    def with_layouts(
        self, layout_a: Layout, layout_b: Layout, layout_c: Layout
    ) -> "GemmScan":
        """Set the matrix layouts for A, B, and C."""
        self.layout_a = layout_a
        self.layout_b = layout_b
        self.layout_c = layout_c
        return self

    def with_scaling(self, alpha: float = 1.0, beta: float = 0.0) -> "GemmScan":
        """Set the scaling factors alpha and beta."""
        self.alpha = alpha
        self.beta = beta
        return self

    def output_to(self, directory: Union[str, Path]) -> "GemmScan":
        """Set the output directory for results."""
        self.output_dir = Path(directory)
        return self

    def growing_with_respect_to(self, dimension: str) -> "GemmScan":
        """
        Configure the scan to use one dimension as the 'growing' dimension,
        with other dimensions computed in relation to it.

        Args:
            dimension: The dimension that will be used as the reference ("M", "N", or "K")

        Returns:
            Self for method chaining

        Example:
            scanner.growing_with_respect_to("M")
                  .with_n_equals(lambda m: m * 256)
                  .with_k_equals(lambda m: m * 256)
        """
        if dimension not in ["M", "N", "K"]:
            raise ValueError(
                f"Growing dimension must be 'M', 'N', or 'K', got {dimension}"
            )

        self.growing_dimension = dimension
        return self

    def with_m_equals(self, func: Callable[[int], int]) -> "GemmScan":
        """Define how M should be computed based on the growing dimension."""
        if self.growing_dimension == "M":
            raise ValueError(
                "Cannot set a relationship for M when M is the growing dimension"
            )
        self.dimension_relationships["M"] = func
        return self

    def with_n_equals(self, func: Callable[[int], int]) -> "GemmScan":
        """Define how N should be computed based on the growing dimension."""
        if self.growing_dimension == "N":
            raise ValueError(
                "Cannot set a relationship for N when N is the growing dimension"
            )
        self.dimension_relationships["N"] = func
        return self

    def with_k_equals(self, func: Callable[[int], int]) -> "GemmScan":
        """Define how K should be computed based on the growing dimension."""
        if self.growing_dimension == "K":
            raise ValueError(
                "Cannot set a relationship for K when K is the growing dimension"
            )
        self.dimension_relationships["K"] = func
        return self

    def _generate_test_cases(self) -> Iterator[GemmTestCase]:
        """Generate all test cases to run for this scan."""
        if self.growing_dimension:
            # Generate test cases based on the growing dimension
            growing_dim = self.growing_dimension
            for data_type in self.data_types:
                if growing_dim == "M":
                    for m in self.m_values:
                        # Compute other dimensions based on M
                        n = self.dimension_relationships["N"](m)
                        if "K" in self.dimension_relationships:
                            k = self.dimension_relationships["K"](m)
                        else:
                            k = n if self.nk_linked else self.k_values[0]
                        yield GemmTestCase(
                            data_type=data_type,
                            m=m,
                            n=n,
                            k=k,
                        )
                elif growing_dim == "N":
                    for n in self.n_values:
                        # Compute other dimensions based on N
                        m = self.dimension_relationships["M"](n)
                        if "K" in self.dimension_relationships:
                            k = self.dimension_relationships["K"](n)
                        else:
                            k = n if self.nk_linked else self.k_values[0]
                        yield GemmTestCase(
                            data_type=data_type,
                            m=m,
                            n=n,
                            k=k,
                        )
                elif growing_dim == "K":
                    for k in self.k_values:
                        # Compute other dimensions based on K
                        m = self.dimension_relationships["M"](k)
                        n = self.dimension_relationships["N"](k)
                        yield GemmTestCase(
                            data_type=data_type,
                            m=m,
                            n=n,
                            k=k,
                        )
        else:
            # Generate all combinations
            for data_type in self.data_types:
                for m in self.m_values:
                    for n in self.n_values:
                        k_vals = [n] if self.nk_linked else self.k_values
                        for k in k_vals:
                            yield GemmTestCase(
                                data_type=data_type,
                                m=m,
                                n=n,
                                k=k,
                            )

    def _create_kernel_spec(self, test_case: GemmTestCase) -> GemmKernelSpec:
        """Create a kernel specification for a test case."""
        try:
            log.debug(f"Creating kernel spec for {test_case.name}")

            # Create a GemmKernelSpecBuilder
            builder = GemmKernelSpecBuilder()

            # Set the data type
            builder.data_type(test_case.data_type)

            # Create GemmParams
            params = GemmParams(
                m=test_case.m,
                n=test_case.n,
                k=test_case.k,
                alpha=self.alpha,
                beta=self.beta,
                layout_a=self.layout_a,
                layout_b=self.layout_b,
                layout_c=self.layout_c,
            )

            # Set the operation parameters
            builder.operation_params(params)

            # Create and set input tensors
            # Matrix A: M x K
            a_tensor = TensorSpec.create_2d(
                rows=test_case.m,
                cols=test_case.k,
                layout=self.layout_a,
                data_type=test_case.data_type,
            )

            # Matrix B: K x N
            b_tensor = TensorSpec.create_2d(
                rows=test_case.k,
                cols=test_case.n,
                layout=self.layout_b,
                data_type=test_case.data_type,
            )

            # Set inputs
            builder.inputs(a=a_tensor, b=b_tensor)

            # Create and set output tensor
            # Matrix C: M x N
            c_tensor = TensorSpec.create_2d(
                rows=test_case.m,
                cols=test_case.n,
                layout=self.layout_c,
                data_type=test_case.data_type,
            )

            # Set outputs
            builder.outputs(c=c_tensor)

            # Build the kernel spec
            kernel_spec = builder.build()
            return kernel_spec

        except Exception as e:
            log.error(f"Error creating kernel spec for {test_case.name}: {e}")
            raise

    def run(self) -> Dict[str, List[Any]]:
        """
        Run the GEMM scan with the configured parameters.

        Returns:
            Dictionary mapping data type names to lists of profile results
        """
        self._validate_config()
        self._setup_directories()

        # Generate all test cases
        test_cases = list(self._generate_test_cases())
        total_cases = len(test_cases)
        log.info(f"Generated {total_cases} test cases")

        # Initialize results dictionary
        self._results = {dt.name: [] for dt in self.data_types}

        # Log scan configuration
        log.info("Starting GEMM performance scan...")
        log.info(f"Data types: {[dt.name for dt in self.data_types]}")

        if self.growing_dimension:
            log.info(f"Growing dimension: {self.growing_dimension}")
            # Additional logging for the growing dimension configuration
            if self.growing_dimension == "M":
                sample_values = self.m_values[:3]  # Show first 3 values as a sample
                log.info(
                    f"M values (sample): {sample_values}{'...' if len(self.m_values) > 3 else ''}"
                )
            elif self.growing_dimension == "N":
                sample_values = self.n_values[:3]
                log.info(
                    f"N values (sample): {sample_values}{'...' if len(self.n_values) > 3 else ''}"
                )
            elif self.growing_dimension == "K":
                sample_values = self.k_values[:3]
                log.info(
                    f"K values (sample): {sample_values}{'...' if len(self.k_values) > 3 else ''}"
                )
        else:
            log.info(f"M values: {self.m_values}")
            log.info(f"N values: {self.n_values}")
            log.info(f"K values: {'Same as N' if self.nk_linked else self.k_values}")

        log.info(f"Output directory: {self._base_output_dir}")

        # Run all test cases
        for i, test_case in enumerate(test_cases, 1):
            output_file = self._base_output_dir / test_case.output_path
            # Ensure output directory exists
            output_dir = output_file.parent
            if not output_dir.exists():
                log.warning(
                    f"Output directory does not exist, creating it now: {output_dir}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"[{i}/{total_cases}] Testing {test_case.name}")

            try:
                # Create kernel spec
                kernel_spec = self._create_kernel_spec(test_case)

                profiler = Profiler(config=self.profile_config, kernel_spec=kernel_spec)

                # Run profiling
                result = profiler.profile_with_engine(
                    engine_type=self.engine_type,
                    warmup_iterations=self.warmup_iterations,
                    output_file=output_file,
                )

                # Store result
                self._results[test_case.data_type.name].append(result)

                log.info(f"  Profiling successful, results saved to: {output_file}")

            except Exception as e:
                log.error(f"  ✗ Failed for {test_case.name}: {e}")
                raise e

        return self._results

    def _validate_config(self) -> None:
        """Validate the configuration before running the scan."""
        if not self.engine_type:
            raise ValueError("Engine type must be set using with_engine_type()")

        if not self.data_types:
            raise ValueError(
                "At least one data type must be specified using with_data_types()"
            )

        if not self.m_values:
            raise ValueError("M values must be specified using for_m_values()")

        if not self.n_values:
            raise ValueError("N values must be specified using for_n_values()")

        if not self.nk_linked and not self.k_values:
            raise ValueError(
                "K values must be specified using for_k_values() or with_k_equals_n()"
            )

        # Validate growing dimension configuration
        if self.growing_dimension:
            # Check that we have values for the growing dimension
            if self.growing_dimension == "M" and not self.m_values:
                raise ValueError(
                    f"No M values specified for growing dimension. Got {self.m_values}"
                )
            elif self.growing_dimension == "N" and not self.n_values:
                raise ValueError(
                    f"No N values specified for growing dimension. Got {self.n_values}"
                )
            elif self.growing_dimension == "K" and not self.k_values:
                raise ValueError(
                    f"No K values specified for growing dimension. Got {self.k_values}"
                )

            # Check that we have relationship functions for the other dimensions
            required_dimensions = {"M", "N", "K"} - {self.growing_dimension}
            missing_dimensions = required_dimensions - set(
                self.dimension_relationships.keys()
            )

            if missing_dimensions:
                raise ValueError(
                    f"Missing relationship functions for dimensions: {missing_dimensions}. "
                    f"When using growing_with_respect_to('{self.growing_dimension}'), "
                    f"you must define relationships for all other dimensions."
                )

    def _setup_directories(self) -> None:
        """Set up output directories for the scan."""
        output_dir = Path(self.output_dir)

        # Create main output directory
        engine_name = self.engine_type.name.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._base_output_dir = output_dir / f"gemm_{engine_name}_{timestamp}"

        if self._base_output_dir.exists():
            log.warning(f"Output directory already exists: {self._base_output_dir}")
        else:
            log.info(f"Creating output directory: {self._base_output_dir}")
            self._base_output_dir.mkdir(parents=True, exist_ok=True)

        # Create plots directory
        self._plots_dir = self._base_output_dir / "plots"
        self._plots_dir.mkdir(exist_ok=True)


class GemmPlotter(OperationPlotter):
    """
    Concrete implementation of OperationPlotter for GEMM operations.
    """

    @classmethod
    def get_operation_type(cls) -> OperationType:
        """Get the operation type for GEMM operations."""
        return OperationType.GEMM

    @classmethod
    def get_default_hover_data(cls) -> Dict[str, bool]:
        """Get the default hover data configuration for GEMM operations."""
        return {
            "time_scaled": False,
            "time_ms": True,
            "M": True,
            "N": True,
            "K": True,
        }

    @classmethod
    def get_default_group_by(cls) -> str:
        """Get the default column to group by for GEMM operations."""
        return "M"

    @classmethod
    def calculate_roofline_data(cls, result_set: ProfileResultSet) -> "pl.DataFrame":
        """
        Calculate data needed for roofline model visualization for GEMM operations.

        Args:
            result_set: ProfileResultSet containing GEMM profiling results
            precision: Precision format to use for peak performance
            compute_unit: Unit of performance measure ('tflops' or 'gflops')

        Returns:
            DataFrame with additional columns for roofline analysis
        """
        # Check if there are results to process
        if not result_set.results:
            log.warning("No results in result_set")
            return pl.DataFrame()

        kernel_spec = result_set.kernel_spec
        operation_params = kernel_spec.operation_params

        accelerator_spec = result_set.accelerator_spec

        # Get peak performance values from accelerator specs
        peak_compute = accelerator_spec.get_peak_compute(
            kernel_spec.data_type
        ).to_giga()
        peak_bandwidth = accelerator_spec.peak_bandwidth
        arithmetic_intensity = calculate_arithmetic_intensity(
            operation_params, kernel_spec.data_type
        )

        # Get the dataframe from the result_set
        df = result_set.results_as_dataframe

        if len(df) == 0:
            log.warning("Empty dataframe in result_set")
            return pl.DataFrame()

        memory_constraint = GigaBytesPerSecond(
            peak_bandwidth.value * arithmetic_intensity.value
        )
        df = df.with_columns(
            [
                pl.lit(operation_params.m).alias("m"),
                pl.lit(operation_params.n).alias("n"),
                pl.lit(operation_params.k).alias("k"),
                pl.lit(kernel_spec.data_type).alias("data_type"),
                pl.lit(peak_compute.value).alias("peak_compute_value"),
                pl.lit(peak_compute.symbol).alias("peak_compute_unit"),
                pl.lit(peak_bandwidth.value).alias("peak_bandwidth_value"),
                pl.lit(peak_bandwidth.symbol).alias("peak_bandwidth_unit"),
                pl.lit(arithmetic_intensity.value).alias("arithmetic_intensity_value"),
                pl.lit(arithmetic_intensity.symbol).alias("arithmetic_intensity_unit"),
                pl.lit(memory_constraint.value).alias("memory_constraint_value"),
                pl.lit(memory_constraint.symbol).alias("memory_constraint_unit"),
            ]
        )

        # Create a unique group identifier for each configuration
        df = df.with_columns(
            [
                (
                    pl.col("m").cast(pl.Utf8)
                    + "_"
                    + pl.col("n").cast(pl.Utf8)
                    + "_"
                    + pl.col("k").cast(pl.Utf8)
                ).alias("group")
            ]
        )

        # Calculate attainable performance (min of compute peak and memory constraint)
        df = df.with_columns(
            [
                pl.min_horizontal(
                    pl.col("peak_compute_value"), pl.col("memory_constraint_value")
                ).alias("attainable_performance_value")
            ]
        )

        # Calculate memory constraint line using units
        # df = df.with_columns(
        #     [
        #         (peak_bandwidth * pl.col("arithmetic_intensity")).alias(
        #             "memory_constraint"
        #         )
        #     ]
        # )

        # Calculate arithmetic intensity using the function from operations/gemm.py
        # We use map_elements to apply it to each row in a struct
        # df = df.with_columns(
        #     [
        #         pl.struct(["m", "n", "k", "data_type"])
        #         .map_elements(
        #             lambda row: calculate_arithmetic_intensity(
        #                 GemmParams(
        #                     m=row["m"],
        #                     n=row["n"],
        #                     k=row["k"],
        #                 ),
        #                 kernel_spec.data_type,
        #             ),
        #             return_dtype=pl.Float64,
        #         )
        #         .alias("arithmetic_intensity")
        #     ]
        # )
        # df = df.with_columns(
        #     [
        #         pl.col("compute_performance")
        #     ]
        # )

        # # Handle time metrics
        # if "latency" in df.columns:
        #     # Convert microseconds to milliseconds for visualization
        #     df = df.with_columns(
        #         [
        #             pl.col("latency")
        #             .map_elements(
        #                 lambda us: Microsecond(us).to(Millisecond).base_value,
        #                 return_dtype=pl.Float64,
        #             )
        #             .alias("time_ms"),
        #             (
        #                 pl.col("latency")
        #                 .map_elements(
        #                     lambda us: Microsecond(us).to(Millisecond).base_value,
        #                     return_dtype=pl.Float64,
        #                 )
        #                 .sqrt()
        #             ).alias("time_scaled"),
        #         ]
        #     )
        # elif "time_ms" in df.columns:
        #     df = df.with_columns([(pl.col("time_ms").sqrt()).alias("time_scaled")])
        # elif "avg_kernel_time_ms" in df.columns:
        #     df = df.with_columns(
        #         [
        #             pl.col("avg_kernel_time_ms").alias("time_ms"),
        #             (pl.col("avg_kernel_time_ms").sqrt()).alias("time_scaled"),
        #         ]
        #     )

        # # Add precision information
        # df = df.with_columns(
        #     [
        #         pl.lit(precision.name).alias("precision_format"),
        #         pl.lit(peak_compute).alias("peak_compute"),
        #         pl.lit(peak_bandwidth).alias("peak_bandwidth"),
        #     ]
        # )

        # log.debug(f"Final dataframe has {len(df)} rows")
        # if len(df) > 0:
        #     log.debug(
        #         f"Sample arithmetic intensity values: {df['arithmetic_intensity'].to_list()[:3]}"
        #     )
        #     if "tflops" in df.columns:
        #         log.debug(f"Sample tflops values: {df['tflops'].to_list()[:3]}")
        #     elif "gflops" in df.columns:
        #         log.debug(f"Sample gflops values: {df['gflops'].to_list()[:3]}")

        return df


## should be at the end of the file
register_operation_builder(OperationType.GEMM, GemmKernelSpecBuilder)
