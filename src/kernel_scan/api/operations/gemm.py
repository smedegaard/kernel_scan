"""
GEMM operations implementation.

This module implements the General Matrix Multiplication (GEMM) operations
for the kernel_scan library. It provides the necessary functionality to
define and validate GEMM operations for GPU profiling.
"""

import itertools
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


def calculate_flops(params: GemmParams) -> int:
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

    return flops


def calculate_bytes_moved(params: GemmParams, dtype_size: int) -> int:
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

    return a_bytes + b_bytes + c_read_bytes + c_bytes


def calculate_arithmetic_intensity(params: GemmParams, dtype: DataType) -> float:
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

    return flops / bytes_moved


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
        return self.gemm_params

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


@dataclass
class GemmScanConfig:
    """Configuration parameters for GEMM scanning."""

    engine_type: Optional[EngineType] = None
    data_types: List[DataType] = None
    n_values: List[int] = None
    m_values: List[int] = None
    k_values: List[int] = None
    nk_linked: bool = False  # When True, N=K in all test cases    static_nk: bool = False  # When True, N and K are fixed to static_nk_value

    # Growing dimension configuration
    growing_dimension: Optional[str] = (
        None  # The dimension that grows: "M", "N", or "K"
    )
    dimension_relationships: Dict[str, Callable[[int], int]] = (
        None  # Functions to compute other dimensions
    )

    iterations: int = 10
    warmup_iterations: int = 5
    layout_a: Layout = Layout.ROW_MAJOR
    layout_b: Layout = Layout.ROW_MAJOR
    layout_c: Layout = Layout.ROW_MAJOR
    alpha: float = 1.0
    beta: float = 0.0
    output_dir: Path = Path("results")
    timestamp_format: str = "%Y%m%d_%H%M%S"

    def __post_init__(self):
        """Initialize default values for lists."""
        if self.data_types is None:
            self.data_types = []
        if self.n_values is None:
            self.n_values = []
        if self.m_values is None:
            self.m_values = []
        if self.k_values is None:
            self.k_values = []
        if self.dimension_relationships is None:
            self.dimension_relationships = {}


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

    def __init__(self):
        """Initialize the GEMM scanner with default configuration."""
        # lazy import to avoid circular imports
        from kernel_scan.api.profiler import Profiler

        self.config = GemmScanConfig()
        self.profiler = Profiler()
        self._results = {}
        self._base_output_dir = None
        self._plots_dir = None

    def with_engine_type(self, engine_type: EngineType) -> "GemmScan":
        """
        Set the engine type to use for profiling.
        This builder method is mandatory before calling run().
        """
        self.config.engine_type = engine_type
        return self

    def with_data_types(self, data_types: List[DataType]) -> "GemmScan":
        """Set the data types to scan."""
        self.config.data_types = data_types
        return self

    def for_n_values(self, n_values: List[int]) -> "GemmScan":
        """Set the N dimension values to scan."""
        self.config.n_values = n_values
        return self

    def for_m_values(self, m_values: List[int]) -> "GemmScan":
        """Set the M dimension values to scan."""
        self.config.m_values = m_values
        return self

    def for_k_values(self, k_values: List[int]) -> "GemmScan":
        """Set the K dimension values to scan."""
        self.config.k_values = k_values
        return self

    def with_k_equals_n(self) -> "GemmScan":
        """Configure the scan to use K = N for all test cases."""
        self.config.nk_linked = True
        return self

    def iterations(self, count: int) -> "GemmScan":
        """Set the number of profiling iterations."""
        self.config.iterations = count
        return self

    def warmup(self, count: int) -> "GemmScan":
        """Set the number of warmup iterations."""
        self.config.warmup_iterations = count
        return self

    def with_layouts(
        self, layout_a: Layout, layout_b: Layout, layout_c: Layout
    ) -> "GemmScan":
        """Set the matrix layouts for A, B, and C."""
        self.config.layout_a = layout_a
        self.config.layout_b = layout_b
        self.config.layout_c = layout_c
        return self

    def with_scaling(self, alpha: float = 1.0, beta: float = 0.0) -> "GemmScan":
        """Set the scaling factors alpha and beta."""
        self.config.alpha = alpha
        self.config.beta = beta
        return self

    def output_to(self, directory: Union[str, Path]) -> "GemmScan":
        """Set the output directory for results."""
        self.config.output_dir = Path(directory)
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

        self.config.growing_dimension = dimension
        return self

    def with_m_equals(self, func: Callable[[int], int]) -> "GemmScan":
        """Define how M should be computed based on the growing dimension."""
        if self.config.growing_dimension == "M":
            raise ValueError(
                "Cannot set a relationship for M when M is the growing dimension"
            )
        self.config.dimension_relationships["M"] = func
        return self

    def with_n_equals(self, func: Callable[[int], int]) -> "GemmScan":
        """Define how N should be computed based on the growing dimension."""
        if self.config.growing_dimension == "N":
            raise ValueError(
                "Cannot set a relationship for N when N is the growing dimension"
            )
        self.config.dimension_relationships["N"] = func
        return self

    def with_k_equals(self, func: Callable[[int], int]) -> "GemmScan":
        """Define how K should be computed based on the growing dimension."""
        if self.config.growing_dimension == "K":
            raise ValueError(
                "Cannot set a relationship for K when K is the growing dimension"
            )
        self.config.dimension_relationships["K"] = func
        return self

    def _generate_test_cases(self) -> Iterator[GemmTestCase]:
        """Generate all test cases as an iterator."""
        test_cases = []

        # Handle growing dimension case
        if self.config.growing_dimension:
            dimension = self.config.growing_dimension

            # Get the values for the growing dimension
            if dimension == "M":
                growing_values = self.config.m_values
            elif dimension == "N":
                growing_values = self.config.n_values
            elif dimension == "K":
                growing_values = self.config.k_values

            log.info(
                f"Generating test cases for growing dimension '{dimension}' with values: {growing_values}"
            )

            # Check that we have relationship functions for the other dimensions
            required_dimensions = {"M", "N", "K"} - {dimension}
            missing_dimensions = required_dimensions - set(
                self.config.dimension_relationships.keys()
            )

            if missing_dimensions:
                raise ValueError(
                    f"Missing relationship functions for dimensions: {missing_dimensions}. "
                    f"When using growing_with_respect_to('{dimension}'), you must define "
                    f"relationships for all other dimensions."
                )

            # For each data_type and growing value, compute the other dimensions
            for data_type in self.config.data_types:
                for value in growing_values:
                    # Compute values for other dimensions
                    m = (
                        value
                        if dimension == "M"
                        else self.config.dimension_relationships["M"](value)
                    )
                    n = (
                        value
                        if dimension == "N"
                        else self.config.dimension_relationships["N"](value)
                    )
                    k = (
                        value
                        if dimension == "K"
                        else self.config.dimension_relationships["K"](value)
                    )

                    test_case = GemmTestCase(data_type=data_type, m=m, n=n, k=k)
                    test_cases.append(test_case)

        # Handle existing cases
        elif self.config.nk_linked:
            # For each (data_type, n, m) combination, use k=n
            for data_type, n, m in itertools.product(
                self.config.data_types, self.config.n_values, self.config.m_values
            ):
                test_case = GemmTestCase(data_type=data_type, m=m, n=n, k=n)
                test_cases.append(test_case)
        else:
            # For each (data_type, n, m, k) combination
            for data_type, n, m, k in itertools.product(
                self.config.data_types,
                self.config.n_values,
                self.config.m_values,
                self.config.k_values,
            ):
                test_case = GemmTestCase(data_type=data_type, m=m, n=n, k=k)
                test_cases.append(test_case)

        # Log the comprehensive test case matrix
        self._log_test_case_matrix(test_cases)

        # Yield all test cases
        for test_case in test_cases:
            yield test_case

    def _log_test_case_matrix(self, test_cases: List[GemmTestCase]) -> None:
        """Log a comprehensive matrix of all generated test cases."""
        if not test_cases:
            log.info("No test cases generated.")
            return

        # Group test cases by data type
        test_cases_by_data_type = {}
        for test_case in test_cases:
            data_type_name = test_case.data_type.name
            if data_type_name not in test_cases_by_data_type:
                test_cases_by_data_type[data_type_name] = []
            test_cases_by_data_type[data_type_name].append(test_case)

        # Log summary statistics
        total_cases = len(test_cases)
        data_types = list(test_cases_by_data_type.keys())

        log.info(f"\n{'=' * 80}")
        log.info("GEMM TEST CASE MATRIX SUMMARY")
        log.info(f"{'=' * 80}")
        log.info(f"Total test cases: {total_cases}")
        log.info(f"Data types: {', '.join(data_types)}")
        log.info(
            f"Cases per data type: {total_cases // len(data_types) if data_types else 0}"
        )

        # Log detailed matrix for each data type
        for data_type_name, cases in test_cases_by_data_type.items():
            log.info(f"\n{'-' * 60}")
            log.info(f"TEST CASES FOR {data_type_name}")
            log.info(f"{'-' * 60}")

            # Create a table header
            log.info(
                f"{'Index':<6} {'M':<8} {'N':<8} {'K':<8} {'FLOPS':<12} {'Matrix Size'}"
            )
            log.info(f"{'-' * 60}")

            # Log each test case with calculated metrics
            for i, case in enumerate(cases, 1):
                flops = 2 * case.m * case.n * case.k  # Basic GEMM FLOPS calculation
                matrix_size = f"{case.m}×{case.n}×{case.k}"
                log.info(
                    f"{i:<6} {case.m:<8} {case.n:<8} {case.k:<8} {flops:<12,} {matrix_size}"
                )

            # Log statistics for this data type
            m_values = sorted(set(case.m for case in cases))
            n_values = sorted(set(case.n for case in cases))
            k_values = sorted(set(case.k for case in cases))

            log.info(f"\nStatistics for {data_type_name}:")
            log.info(f"  M values: {m_values}")
            log.info(f"  N values: {n_values}")
            log.info(f"  K values: {k_values}")
            log.info(f"  Total cases: {len(cases)}")

            # Calculate total FLOPS for this data type
            total_flops = sum(2 * case.m * case.n * case.k for case in cases)
            log.info(f"  Total FLOPS: {total_flops:,}")

        log.info(f"\n{'=' * 80}")
        log.info("END OF TEST CASE MATRIX")
        log.info(f"{'=' * 80}\n")

    def _create_kernel_spec(self, test_case: GemmTestCase) -> KernelSpec:
        """Create a kernel specification for a test case."""
        return (
            KernelSpec.builder()
            .operation_type(OperationType.GEMM)
            .data_type(test_case.data_type)
            .operation_params(
                GemmParams(
                    m=test_case.m,
                    n=test_case.n,
                    k=test_case.k,
                    alpha=self.config.alpha,
                    beta=self.config.beta,
                    layout_a=self.config.layout_a,
                    layout_b=self.config.layout_b,
                    layout_c=self.config.layout_c,
                )
            )
            .inputs(
                a=TensorSpec.create_2d(
                    test_case.m, test_case.k, self.config.layout_a, test_case.data_type
                ),
                b=TensorSpec.create_2d(
                    test_case.k, test_case.n, self.config.layout_b, test_case.data_type
                ),
            )
            .outputs(
                c=TensorSpec.create_2d(
                    test_case.m, test_case.n, self.config.layout_c, test_case.data_type
                )
            )
            .iterations(self.config.iterations)
            .name(test_case.name)
            .build()
        )

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
        self._results = {dt.name: [] for dt in self.config.data_types}

        # Log scan configuration
        log.info("Starting GEMM performance scan...")
        log.info(f"Data types: {[dt.name for dt in self.config.data_types]}")

        if self.config.growing_dimension:
            log.info(f"Growing dimension: {self.config.growing_dimension}")

            # Log a sample of dimension values for clarity
            dimension = self.config.growing_dimension
            values = []
            if dimension == "M":
                values = self.config.m_values[:3]  # Show first 3 values as a sample
            elif dimension == "N":
                values = self.config.n_values[:3]
            elif dimension == "K":
                values = self.config.k_values[:3]

            log.info(
                f"{dimension} values (sample): {values}{'...' if len(values) < 3 else ''}"
            )
            log.info("Other dimensions will be computed based on relationships")
        else:
            log.info(f"N values: {self.config.n_values}")
            log.info(f"M values: {self.config.m_values}")
            log.info(
                f"K values: {'Same as N' if self.config.nk_linked else self.config.k_values}"
            )
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
                # Create kernel spec and run profiling
                kernel_spec = self._create_kernel_spec(test_case)
                result = self.profiler.profile_with_engine(
                    kernel_spec,
                    self.config.engine_type,
                    warmup_iterations=self.config.warmup_iterations,
                    output_file=str(output_file),
                )

                # Store result
                self._results[test_case.data_type.name].append(result)

                log.info(f"  Profiling successful, results saved to: {output_file}")

            except Exception as e:
                log.error(f"  ✗ Failed for {test_case.name}: {e}")
                raise e

        return self._results

    def _validate_config(self):
        """Validate that the configuration is complete and consistent."""
        if self.config.engine_type is None:
            raise ValueError("Engine type must be specified using with_engine_type()")

        if not self.config.data_types:
            raise ValueError(f"No data types specified. Got {self.config.data_types}")

        # Validate based on configuration mode
        if self.config.growing_dimension:
            # Check that we have values for the growing dimension
            if self.config.growing_dimension == "M" and not self.config.m_values:
                raise ValueError(
                    f"No M values specified for growing dimension. Got {self.config.m_values}"
                )
            elif self.config.growing_dimension == "N" and not self.config.n_values:
                raise ValueError(
                    f"No N values specified for growing dimension. Got {self.config.n_values}"
                )
            elif self.config.growing_dimension == "K" and not self.config.k_values:
                raise ValueError(
                    f"No K values specified for growing dimension. Got {self.config.k_values}"
                )

            # Check that we have relationship functions for the other dimensions
            required_dimensions = {"M", "N", "K"} - {self.config.growing_dimension}
            missing_dimensions = required_dimensions - set(
                self.config.dimension_relationships.keys()
            )

            if missing_dimensions:
                raise ValueError(
                    f"Missing relationship functions for dimensions: {missing_dimensions}. "
                    f"When using growing_with_respect_to('{self.config.growing_dimension}'), "
                    f"you must define relationships for all other dimensions."
                )
        else:
            if not self.config.m_values:
                raise ValueError(f"No M values specified. Got {self.config.m_values}")

            if not self.config.n_values:
                raise ValueError(f"No N values specified. Got {self.config.n_values}")

            if not self.config.nk_linked and not self.config.k_values:
                raise ValueError(
                    f"No K values specified and K ≠ N. Got {self.config.k_values}"
                )

    def _setup_directories(self):
        """Set up the output directories for the scan."""
        timestamp = datetime.now().strftime(self.config.timestamp_format)
        self._base_output_dir = self.config.output_dir / f"gemm_scan_{timestamp}"
        self._plots_dir = self._base_output_dir / "plots"

        # Create the base and plots directories
        self._base_output_dir.mkdir(parents=True, exist_ok=True)
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Created base output directory: {self._base_output_dir}")
        log.info(f"Created plots directory: {self._plots_dir}")

        # Generate all test cases to identify all needed subdirectories
        test_cases = list(self._generate_test_cases())
        log.info(f"Setting up directories for {len(test_cases)} test cases")

        # Create subdirectories for each data type and NK value
        created_dirs = set()  # Track directories we've created to avoid duplicate logs

        for test_case in test_cases:
            # Create directory for each data type
            data_type_dir = self._base_output_dir / test_case.data_type.name
            if str(data_type_dir) not in created_dirs:
                data_type_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(str(data_type_dir))
                log.info(
                    f"Created directory for {test_case.data_type.name}: {data_type_dir}"
                )

            # Create directory for each NK value
            nk_dir = data_type_dir / f"NK{test_case.n}"
            if str(nk_dir) not in created_dirs:
                nk_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(str(nk_dir))
                log.info(f"Created directory for NK{test_case.n}: {nk_dir}")


register_operation_builder(OperationType.GEMM, GemmKernelSpecBuilder)
