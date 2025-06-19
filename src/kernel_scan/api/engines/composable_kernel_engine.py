"""
Composable Kernel Engine implementation.

This module provides an implementation of the ComputeEngine interface
that interacts with AMD's Composable Kernel library via the ckProfiler tool.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from kernel_scan.api.operations.gemm import (
    GemmInputs,
    GemmOperationParams,
    GemmOutputs,
    GemmParams,
)
from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.engine import ComputeEngine
from kernel_scan.core.results import ProfileResult, ProfileResultSet
from kernel_scan.core.specs import AcceleratorSpec, KernelSpec, TensorSpec
from kernel_scan.core.types import (
    DataType,
    Layout,
    OperationType,
)

log = logging.getLogger(__name__)


class ComposableKernelEngine(ComputeEngine):
    """
    ComputeEngine implementation using AMD's Composable Kernel library.

    This engine uses the ckProfiler tool to profile GEMM operations.
    """

    def __init__(
        self,
        config: Optional[ProfileConfig] = None,
        accelerator_specs: Optional[AcceleratorSpec] = None,
    ):
        """
        Initialize the Composable Kernel engine.

        Args:
            config: Optional configuration for the engine
            accelerator_specs: Optional accelerator specifications for the engine
        """
        super().__init__(config, accelerator_specs)
        self._profiler_path = None

    def initialize(self) -> bool:
        """
        Initialize the engine and locate the ckProfiler executable.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self._initialized:
            return True

        # Try to find ckProfiler executable
        profiler_path = self._find_profiler(self.config.custom_profiler_path)
        if not profiler_path:
            # Try user-provided path in config
            if "profiler_path" in self.config.engine_config:
                profiler_path = self.config.engine_config["profiler_path"]
                if not os.path.exists(profiler_path):
                    raise FileNotFoundError(f"ckProfiler not found at {profiler_path}")
            else:
                raise FileNotFoundError(
                    "ckProfiler executable not found. Please install Composable Kernel "
                    "or specify its path in the engine configuration."
                )

        self._profiler_path = profiler_path
        self._initialized = True

        # Set accelerator specs if not provided
        if self._accelerator_specs is None:
            # Auto-detect hardware
            self._accelerator_specs = AcceleratorSpec.detect_hardware()
            log.info(f"Detected hardware: {self._accelerator_specs.name}")
        else:
            log.info(f"Using provided hardware specs: {self._accelerator_specs.name}")

        return True

    def is_supported(self, kernel_spec: KernelSpec) -> bool:
        """
        Check if the engine supports the given kernel specification.

        Args:
            kernel_spec: The kernel specification to check

        Returns:
            True if the kernel specification is supported
        """
        # Currently, only GEMM operations are supported
        if kernel_spec.operation_type != OperationType.GEMM:
            return False

        # Get GEMM parameters
        gemm_params = kernel_spec.operation_params
        if not isinstance(gemm_params, GemmOperationParams):
            return False

        return True

    def profile(
        self, kernel_spec: KernelSpec, output_file: Optional[str] = None
    ) -> ProfileResultSet:
        """
        Profile the given kernel specification using ckProfiler.

        Args:
            kernel_spec: The kernel specification to profile
            output_file: Optional path to save the raw output (if not provided, uses a temp file)

        Returns:
            ProfileResultSet containing the profiling results

        Raises:
            ValueError: If the kernel specification is not supported
            RuntimeError: If profiling fails
        """
        if not self._initialized:
            self.initialize()

        if not self.is_supported(kernel_spec):
            raise ValueError(f"Kernel specification not supported: {kernel_spec}")

        # Only GEMM operations are currently supported
        if kernel_spec.operation_type != OperationType.GEMM:
            raise ValueError(
                f"Operation type not supported: {kernel_spec.operation_type}"
            )

        # Get GEMM parameters
        gemm_params = kernel_spec.operation_params
        if not isinstance(gemm_params, GemmOperationParams):
            raise ValueError(f"Invalid operation parameters: {gemm_params}")

        params = gemm_params.params

        # Use provided output file or create a temporary one
        temp_file_created = False
        if output_file is None:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
                output_file = tmp_file.name
                temp_file_created = True

        try:
            # Run ckProfiler and get results
            args = self._build_profiler_args(kernel_spec, params, output_file)

            log.info(f"Running ckProfiler with arguments: {' '.join(args)}")
            _result = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
            )

            try:
                profile_data = self._parse_profiler_output(output_file)
            except Exception as e:
                raise RuntimeError(f"Failed to parse profiler output: {e}")

            # Create profile result
            profile_results = [
                self._create_profile_result(kernel_spec, result)
                for result in profile_data
            ]

            # Create result set with the profile results
            result_set = ProfileResultSet(profile_results, self.accelerator_specs)

            # Set engine and hardware info
            result_set.engine_name = "ComposableKernel"
            result_set.engine_info = {"profiler_path": self._profiler_path}
            result_set.hardware_info = self.get_hardware_info()

            # Mark the best result if there are multiple results
            if len(profile_results) > 1:
                result_set.mark_best_results(metric="time_ms", lower_is_better=True)

            return result_set
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ckProfiler failed with exit code {e.returncode}. "
                f"stdout: {e.stdout}, stderr: {e.stderr}"
            )
        finally:
            # Clean up temporary file only if we created it
            if temp_file_created and os.path.exists(output_file):
                os.unlink(output_file)

    def get_available_kernels(self) -> List[Dict[str, Any]]:
        """
        Get a list of available kernels supported by this engine.

        Returns:
            List of dictionaries describing available kernels
        """
        # This would typically query the Composable Kernel library
        # for available kernels, but we'll provide a basic list for now
        return [
            {
                "name": "gemm_ck_default",
                "operation_type": OperationType.GEMM,
                "data_types": [
                    DataType.FLOAT32,
                    DataType.FLOAT16,
                    DataType.BFLOAT16,
                    DataType.INT8,
                ],
                "description": "Default GEMM implementation in Composable Kernel",
            }
        ]

    def _find_profiler(self, path: Optional[str]) -> Optional[str]:
        """
        Find the ckProfiler executable.

        Returns:
            Path to the ckProfiler executable or None if not found
        """
        # Try common locations
        common_paths = [
            "./bin/ckProfiler",
            "~/bin/ckProfiler",
            "/opt/rocm/bin/ckProfiler",
            "/usr/local/bin/ckProfiler",
        ]

        if path:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path

        for path in common_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path

        # Try to find in PATH
        try:
            result = subprocess.run(
                ["which", "ckProfiler"],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return None

    def _build_profiler_args(
        self, kernel_spec: KernelSpec, params: GemmParams, output_file: str
    ) -> List[str]:
        """
        Build the arguments for the ckProfiler command.

        Help message from `ckProfiler gemm`:

        arg1: tensor operation (gemm: GEMM)
        arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8; 4: fp8)
        arg3: matrix layout (0: A[m, k] * B[k, n] = C[m, n];
                            1: A[m, k] * B[n, k] = C[m, n];
                            2: A[k, m] * B[k, n] = C[m, n];
                            3: A[k, m] * B[n, k] = C[m, n])
        arg4: verification (0: no; 1: yes)
        arg5: initialization (0: no init; 1: integer value; 2: decimal value)
        arg6: print tensor value (0: no; 1: yes)
        arg7: time kernel (0: no, 1: yes)
        arg8 to 13: M, N, K, StrideA, StrideB, StrideC
        optional:
        arg14: number of warm-up cycles (default 1)
        arg15: number of iterations (default 10)

        Args:
            kernel_spec: The kernel specification
            params: The GEMM parameters
            output_file: Path to the output file

        Returns:
            List of command-line arguments
        """
        # Map our data types to ckProfiler data types
        data_type_map = {
            DataType.FLOAT32: "0",  # fp32
            DataType.FLOAT16: "1",  # fp16
            DataType.BFLOAT16: "2",  # bf16
            DataType.INT8: "3",  # int8
        }

        data_type = data_type_map.get(kernel_spec.data_type, "0")

        # Map matrix layout combinations to ckProfiler layout parameter
        layout_map = {
            (Layout.ROW_MAJOR, Layout.ROW_MAJOR): "0",  # A[m,k] * B[k,n]
            (Layout.ROW_MAJOR, Layout.COLUMN_MAJOR): "1",  # A[m,k] * B[n,k]
            (Layout.COLUMN_MAJOR, Layout.ROW_MAJOR): "2",  # A[k,m] * B[k,n]
            (Layout.COLUMN_MAJOR, Layout.COLUMN_MAJOR): "3",  # A[k,m] * B[n,k]
        }

        layout = layout_map.get(
            (params.layout_a, params.layout_b), "0"
        )  # Default to 0 if combination not found

        if (params.layout_a, params.layout_b) not in layout_map:
            log.warning(
                f"Unexpected layout combination: A={params.layout_a}, B={params.layout_b}"
            )

        verify = "0"  # Default to no verification for performance
        if self.config.verify_results:
            verify = "1"

        # Map data types to initialization methods
        # 1: integer initialization, 2: decimal initialization
        # init_map = {
        #     DataType.FLOAT32: "2",  # Decimal for floating point
        #     DataType.FLOAT16: "2",
        #     DataType.BFLOAT16: "2",
        #     DataType.FLOAT64: "2",
        #     DataType.INT8: "1",  # Integer for integer types
        #     DataType.UINT8: "1",
        #     DataType.INT16: "1",
        #     DataType.INT32: "1",
        #     DataType.INT64: "1",
        #     DataType.INT4: "1",
        #     DataType.BOOL: "1",
        # }

        # init = init_map.get(
        #     kernel_spec.data_type, "1"
        # )  # Default to integer init if unknown

        init = "0"

        print_tensor = "0"  # Don't print tensor values
        time_kernel = "1"  # Time the kernel
        warmup = str(self.config.warmup_iterations)
        iterations = str(kernel_spec.iterations)

        # Extract GEMM dimensions
        m = str(params.m)
        n = str(params.n)
        k = str(params.k)

        # todo: take strides as parameters
        # Map layout to stride calculation
        stride_map = {
            # For matrix A
            ("A", Layout.ROW_MAJOR): str(params.k),  # A[m,k] stride for row-major
            ("A", Layout.COLUMN_MAJOR): str(params.m),  # A[k,m] stride for column-major
            # For matrix B
            ("B", Layout.ROW_MAJOR): str(params.n),  # B[k,n] stride for row-major
            ("B", Layout.COLUMN_MAJOR): str(params.k),  # B[n,k] stride for column-major
            # For matrix C
            ("C", Layout.ROW_MAJOR): str(params.n),  # C[m,n] stride for row-major
            ("C", Layout.COLUMN_MAJOR): str(params.m),  # C[n,m] stride for column-major
        }

        stride_a = stride_map[("A", params.layout_a)]
        stride_b = stride_map[("B", params.layout_b)]
        stride_c = stride_map[("C", params.layout_c)]

        # Build the command with the jsonl output format
        args = [
            self._profiler_path,
            "gemm",
            data_type,
            layout,
            verify,
            init,
            print_tensor,
            time_kernel,
            m,
            n,
            k,
            stride_a,
            stride_b,
            stride_c,
            warmup,
            iterations,
            "-o",
            f"jsonl={output_file}",
        ]

        return args

    def _parse_profiler_output(self, output_file: str) -> [Dict[str, Any]]:
        """
        Parse the output file from ckProfiler.

        Args:
            output_file: Path to the output file

        Returns:
            Dictionary containing parsed profiling data
        """
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Profiler output file not found: {output_file}")

        try:
            # Read the JSONL file
            with open(output_file, "r") as f:
                lines = f.readlines()

            if not lines:
                raise ValueError("Profiler output file is empty")

            profile_data = [json.loads(line) for line in lines]

            return profile_data
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Failed to parse profiler output: {e}")

    def _create_profile_result(
        self,
        kernel_spec: KernelSpec,
        profile_data: [Dict[str, Any]],
    ) -> ProfileResult:
        """
        Create a ProfileResult from the parsed profiler output.

        Args:
            kernel_spec: The kernel specification
            profile_data: Parsed profiler output

        Returns:
            ProfileResult containing the profiling results
        """
        # Extract timing information
        kernel_times_ms = []

        # Extract kernel times from the profile data
        # The format depends on the ckProfiler output, this is just an example
        if "execution_time" in profile_data:
            # Convert from microseconds to milliseconds
            times = [float(t) / 1000.0 for t in profile_data["execution_time"]]
            kernel_times_ms.extend(times)
        elif (
            "profiling" in profile_data
            and "execution_time" in profile_data["profiling"]
        ):
            times = [
                float(t) / 1000.0 for t in profile_data["profiling"]["execution_time"]
            ]
            kernel_times_ms.extend(times)

        # Extract metrics
        metrics = {}

        # Extract operation name/description
        operation = profile_data.get("operation", kernel_spec.name or "unnamed")

        # Extract GFLOPS and bandwidth
        if "gflops" in profile_data:
            metrics["tflops"] = (
                float(profile_data["gflops"]) / 1000.0
            )  # Convert to TFLOPS

        # Add bandwidth metric if available
        if "bandwidth" in profile_data:
            metrics["bandwidth"] = float(profile_data["bandwidth"])

        # Create profile result
        return ProfileResult(
            kernel_spec=kernel_spec,
            metrics=metrics,
            operation=operation,
            raw_data=profile_data,
        )

    def _get_data_type_size(self, data_type):
        """Get the size in bytes of a data type."""
        sizes = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.BFLOAT16: 2,
            DataType.FLOAT64: 8,
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.INT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.INT4: 0.5,
            DataType.BOOL: 1,
        }
        return sizes.get(data_type, 4)


class CkProfilerScanner:
    """
    Utility class for scanning GEMM performance across different configurations.

    This class provides a convenient interface for profiling GEMM operations
    across a range of matrix sizes and data types.
    """

    def __init__(
        self,
        engine: Optional[ComposableKernelEngine] = None,
        config: Optional[ProfileConfig] = None,
    ):
        """
        Initialize the scanner.

        Args:
            engine: Optional ComposableKernelEngine instance
            config: Optional ProfileConfig instance
        """
        self.engine = engine or ComposableKernelEngine(config)
        if not isinstance(self.engine, ComposableKernelEngine):
            raise TypeError("Engine must be a ComposableKernelEngine instance")

        self.config = config or ProfileConfig.create_default()
        self.results = ProfileResultSet()
        self.results.engine_name = "ComposableKernel"
        self.engine = ComposableKernelEngine(config)
        self.results.hardware_info = self.engine.get_hardware_info()

    def scan_gemm(
        self,
        m_values: List[int],
        n_values: List[int],
        k_values: List[int],
        data_type: DataType = DataType.FLOAT32,
        layout_a: Layout = Layout.ROW_MAJOR,
        layout_b: Layout = Layout.ROW_MAJOR,
        layout_c: Layout = Layout.ROW_MAJOR,
        alpha: float = 1.0,
        beta: float = 0.0,
        iterations: int = 10,
        output_dir: Optional[str] = None,
    ) -> ProfileResultSet:
        """
        Scan GEMM performance across different matrix sizes.

        Args:
            m_values: List of M dimensions to scan
            n_values: List of N dimensions to scan
            k_values: List of K dimensions to scan
            data_type: Data type for the operation
            layout_a: Memory layout for matrix A
            layout_b: Memory layout for matrix B
            layout_c: Memory layout for matrix C
            alpha: Scalar multiplier for A*B
            beta: Scalar multiplier for C
            iterations: Number of iterations per configuration
            output_dir: Optional directory for saving results

        Returns:
            ProfileResultSet containing all profiling results
        """
        # Initialize the engine if needed
        if not self.engine._initialized:
            self.engine.initialize()

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Count total configurations
        total_configs = len(m_values) * len(n_values) * len(k_values)
        current_config = 0

        print("Starting GEMM performance scan...")
        print(f"M values: {m_values}")
        print(f"N values: {n_values}")
        print(f"K values: {k_values}")
        print(f"Data type: {data_type.name}")
        if output_dir:
            print(f"Output directory: {output_dir}")
        print("")

        # Main scanning loop
        for m in m_values:
            for n in n_values:
                for k in k_values:
                    current_config += 1
                    print(
                        f"[{current_config}/{total_configs}] Testing M={m}, N={n}, K={k}"
                    )

                    # Create kernel specification
                    gemm_params = GemmParams(
                        m=m,
                        n=n,
                        k=k,
                        alpha=alpha,
                        beta=beta,
                        layout_a=layout_a,
                        layout_b=layout_b,
                        layout_c=layout_c,
                    )

                    a_spec = TensorSpec.create_2d(m, k, layout_a, data_type)
                    b_spec = TensorSpec.create_2d(k, n, layout_b, data_type)
                    c_spec = TensorSpec.create_2d(m, n, layout_c, data_type)

                    inputs = GemmInputs(a=a_spec, b=b_spec)
                    outputs = GemmOutputs(c=c_spec)

                    kernel_spec = KernelSpec(
                        operation_type=OperationType.GEMM,
                        data_type=data_type,
                        operation_params=GemmOperationParams(gemm_params),
                        inputs=inputs,
                        outputs=outputs,
                        iterations=iterations,
                        name=f"GEMM_M{m}_N{n}_K{k}_{data_type.name}",
                    )

                    try:
                        # Create output file path if directory is specified
                        output_file = None
                        if output_dir:
                            output_file = os.path.join(
                                output_dir,
                                f"gemm_M{m}_N{n}_K{k}_{data_type.name}.jsonl",
                            )

                        # Profile the kernel with direct output to jsonl file
                        result = self.engine.profile(
                            kernel_spec, output_file=output_file
                        )

                        # Add to results set
                        self.results.add_result(result)

                        if output_file:
                            print(f"  ✓ Success - results saved to {output_file}")
                        else:
                            print(
                                f"  ✓ Success - Avg time: {result.timing.avg_kernel_time_ms:.2f} ms, GFLOPS: {result.metrics.get('gflops', 0):.2f}"
                            )

                    except Exception as e:
                        log.error(f"  ✗ Failed for M={m}, N={n}, K={k}: {e}")
                        raise e

        print("")
        print("Scan completed!")

        return self.results
