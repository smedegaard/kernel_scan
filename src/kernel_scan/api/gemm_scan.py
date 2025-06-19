import datetime
import itertools
import logging
from dataclasses import dataclass
from itertools import Iterator
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

from kernel_scan.spec import GemmParams, KernelSpec, OperationType, TensorSpec
from kernel_scan.types import DataType, EngineType, Layout


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
            .with_engine_type(EngineType.COMPOSABLE_KERNEL)  # Mandatory
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
                    self.config.engine_type,
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
        if self.config.engine_type is None:
            raise ValueError("Engine type must be specified using with_engine_type()")

        if not self.config.data_types:
            raise ValueError(f"No data types specified. Got {self.config.data_types}")

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

        # Create necessary subdirectories in advance
        for data_type in self.config.data_types:
            for n in self.config.n_values:
                (self._base_output_dir / data_type.name / f"NK{n}").mkdir(
                    parents=True, exist_ok=True
                )
