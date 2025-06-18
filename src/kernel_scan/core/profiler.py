"""
Profiler module for kernel profiling.

This module provides the Profiler class, which is the main entry point
for profiling GPU kernels with different engine backends.
"""

import itertools
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Union

from kernel_scan.core.accelerator import AcceleratorSpec
from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.engine import ComputeEngine, EngineType, create_engine
from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.specs import KernelSpec, TensorSpec
from kernel_scan.core.types import (
    DataType,
    GemmParams,
    Layout,
    OperationType,
)

log = logging.getLogger(__name__)


class Profiler:
    """
    Main profiler class for kernel profiling.

    This class provides a simple interface for profiling GPU kernels with
    different engine backends.
    """

    def __init__(
        self,
        config: Optional[ProfileConfig] = None,
        accelerator_specs: Optional[AcceleratorSpec] = None,
    ):
        """
        Initialize a new Profiler instance.

        Args:
            config: Optional configuration for the profiler
        """
        self._config = config or ProfileConfig.create_default()
        self._engines: Dict[EngineType, ComputeEngine] = {}
        self._result_set = ProfileResultSet()
        self._accelerator_specs = accelerator_specs

    @property
    def config(self) -> ProfileConfig:
        """Return the profiler configuration."""
        return self._config

    @property
    def results(self) -> ProfileResultSet:
        """Return the profile result set."""
        return self._result_set

    def profile_with_engine(
        self,
        kernel_spec: KernelSpec,
        engine_type: Union[EngineType, str],
        warmup_iterations: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> ProfileResultSet:
        """
        Profile a kernel with a specific engine.

        Args:
            kernel_spec: The kernel specification to profile
            engine_type: The type of engine to use
            warmup_iterations: Optional number of warmup iterations to run
            output_file: Optional file to save the results to

        Returns:
            ProfileResult containing the profiling results

        Raises:
            ValueError: If the engine type is not supported or the kernel specification is not supported
        """
        # Apply warmup iterations to config if specified
        if warmup_iterations is not None:
            self._config.warmup_iterations = warmup_iterations

        # Get or create the engine
        engine = self._get_engine(
            engine_type, accelerator_specs=self._accelerator_specs
        )

        # Check if the kernel is supported
        if not engine.is_supported(kernel_spec):
            raise ValueError(f"Kernel specification not supported by {engine.name}")

        # Profile the kernel
        result = engine.profile(kernel_spec, output_file=output_file)

        # Save the result to the result set
        self._result_set.add_result(result)

        # Return the result
        return result

    def profile_gemm(
        self,
        m: int,
        n: int,
        k: int,
        data_type,
        engine_type: Union[EngineType, str],
        output_file: str,
        **kwargs,
    ) -> ProfileResultSet:
        """
        Profile a GEMM operation with simplified parameters.

        This is a convenience method for profiling GEMM operations without
        having to create a full KernelSpec object.

        Args:
            m: Number of rows in matrices A and C
            n: Number of columns in matrices B and C
            k: Number of columns in matrix A / rows in matrix B
            data_type: Data type for the operation
            engine_type: The type of engine to use
            **kwargs: Additional parameters for the GEMM operation

        Returns:
            ProfileResult containing the profiling results

        Raises:
            ValueError: If the engine type is not supported or the kernel specification is not supported
        """
        from kernel_scan.core.types import (
            GemmParams,
            Layout,
            OperationType,
            TensorSpec,
        )

        # Extract additional parameters
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 0.0)
        layout_a = kwargs.get("layout_a", Layout.ROW_MAJOR)
        layout_b = kwargs.get("layout_b", Layout.ROW_MAJOR)
        layout_c = kwargs.get("layout_c", Layout.ROW_MAJOR)
        iterations = kwargs.get("iterations", 100)
        warmup_iterations = kwargs.get("warmup_iterations", 10)
        name = kwargs.get("name", f"GEMM_M{m}_N{n}_K{k}")

        # Create a GEMM kernel specification
        kernel_spec = (
            KernelSpec.builder()
            .operation_type(OperationType.GEMM)
            .data_type(data_type)
            .operation_params(
                GemmParams(
                    m=m,
                    n=n,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    layout_a=layout_a,
                    layout_b=layout_b,
                    layout_c=layout_c,
                )
            )
            .inputs(
                a=TensorSpec.create_2d(m, k, layout_a, data_type),
                b=TensorSpec.create_2d(k, n, layout_b, data_type),
            )
            .outputs(c=TensorSpec.create_2d(m, n, layout_c, data_type))
            .iterations(iterations)
            .name(name)
            .build()
        )

        # Profile with the specified engine
        return self.profile_with_engine(
            kernel_spec, engine_type, warmup_iterations, output_file
        )

    def _get_engine(
        self,
        engine_type: Union[EngineType, str],
        accelerator_specs: Optional[AcceleratorSpec],
    ) -> ComputeEngine:
        """
        Get or create an engine instance.

        Args:
            engine_type: The type of engine to get or create

        Returns:
            ComputeEngine instance

        Raises:
            ValueError: If the engine type is not supported
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

        # Create the engine if it doesn't exist
        if engine_type not in self._engines:
            engine = create_engine(engine_type, self._config, self._accelerator_specs)
            engine.initialize()
            self._engines[engine_type] = engine

        return self._engines[engine_type]

    def __del__(self):
        """Clean up resources when the profiler is deleted."""
        for engine in self._engines.values():
            engine.shutdown()
        self._engines.clear()


## Gemm scan
#
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

    data_types: List[DataType] = None
    n_values: List[int] = None
    m_values: List[int] = None
    k_values: List[int] = None
    nk_linked: bool = False  # When True, N=K in all test cases
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
            .iterations(10)
            .warmup(5)
            .run())
    """

    def __init__(self):
        """Initialize the GEMM scanner with default configuration."""
        # lazy import to avoid circular imports
        from kernel_scan.core.profiler import Profiler

        self.config = GemmScanConfig()
        self.profiler = Profiler()
        self._results = {}
        self._base_output_dir = None
        self._plots_dir = None

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

    def _generate_test_cases(self) -> Iterator[GemmTestCase]:
        """Generate all test cases as an iterator using itertools.product."""
        # Determine K values based on configuration
        if self.config.nk_linked:
            # For each (data_type, n, m) combination, use k=n
            for data_type, n, m in itertools.product(
                self.config.data_types, self.config.n_values, self.config.m_values
            ):
                yield GemmTestCase(data_type=data_type, m=m, n=n, k=n)
        else:
            # For each (data_type, n, m, k) combination
            for data_type, n, m, k in itertools.product(
                self.config.data_types,
                self.config.n_values,
                self.config.m_values,
                self.config.k_values,
            ):
                yield GemmTestCase(data_type=data_type, m=m, n=n, k=k)

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
        # Create logger for the GEMM scan

        log = logging.getLogger("kernel_scan.gemm")
        if not log.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            log.addHandler(handler)
            log.setLevel(logging.INFO)

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
        log.info(f"N values: {self.config.n_values}")
        log.info(f"M values: {self.config.m_values}")
        log.info(
            f"K values: {'Same as N' if self.config.nk_linked else self.config.k_values}"
        )
        log.info(f"Output directory: {self._base_output_dir}")

        # Run all test cases
        for i, test_case in enumerate(test_cases, 1):
            output_file = self._base_output_dir / test_case.output_path

            log.info(f"[{i}/{total_cases}] Testing {test_case.name}")

            try:
                # Create kernel spec and run profiling
                kernel_spec = self._create_kernel_spec(test_case)
                result = self.profiler.profile_with_engine(
                    kernel_spec,
                    EngineType.COMPOSABLE_KERNEL,
                    warmup_iterations=self.config.warmup_iterations,
                    output_file=str(output_file),
                )

                # Store result
                self._results[test_case.data_type.name].append(result)

                log.info(f"  Profiling successful, results saved to: {output_file}")

            except Exception as e:
                log.error(f"  ✗ Failed for {test_case.name}: {e}")

        return self._results

    def _validate_config(self):
        """Validate that the configuration is complete and consistent."""
        if not self.config.data_types:
            raise ValueError("No data types specified")

        if not self.config.m_values:
            raise ValueError("No M values specified")

        if not self.config.n_values:
            raise ValueError("No N values specified")

        if not self.config.nk_linked and not self.config.k_values:
            raise ValueError("No K values specified and K ≠ N")

    def _setup_directories(self):
        """Set up the output directories for the scan."""
        timestamp = datetime.now().strftime(self.config.timestamp_format)
        self._base_output_dir = self.config.output_dir / f"gemm_scan_{timestamp}"
        self._plots_dir = self._base_output_dir / "plots"

        # Create the base and plots directories
        self._base_output_dir.mkdir(parents=True, exist_ok=True)
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        # Create necessary subdirectories in advance
        for data_type in self.config.data_types:
            for n in self.config.n_values:
                (self._base_output_dir / data_type.name / f"NK{n}").mkdir(
                    parents=True, exist_ok=True
                )
