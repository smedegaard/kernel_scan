"""
Composable Kernel Engine implementation.

This module provides an implementation of the ComputeEngine interface
that interacts with AMD's Composable Kernel library via the ckProfiler tool.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from kernel_scan.core.config import ProfileConfig
from kernel_scan.core.engine import ComputeEngine
from kernel_scan.core.results import ProfileResult, ProfileResultSet, TimingData
from kernel_scan.core.specs import KernelSpec
from kernel_scan.core.types import (
    DataType,
    GemmInputs,
    GemmOperationParams,
    GemmOutputs,
    GemmParams,
    Layout,
    OperationType,
    TensorSpec,
)


class ComposableKernelEngine(ComputeEngine):
    """
    ComputeEngine implementation using AMD's Composable Kernel library.

    This engine uses the ckProfiler tool to profile GEMM operations.
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        """
        Initialize the Composable Kernel engine.

        Args:
            config: Optional configuration for the engine
        """
        super().__init__(config)
        self._profiler_path = None
        self._hardware_info = {}

    def initialize(self) -> bool:
        """
        Initialize the engine and locate the ckProfiler executable.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self._initialized:
            return True

        # Try to find ckProfiler executable
        profiler_path = self._find_profiler()
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

        # Get hardware information
        self._hardware_info = self._detect_hardware()

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

        # Check data type support
        # The data types supported by ckProfiler may vary
        supported_dtypes = [
            DataType.FLOAT32,
            DataType.FLOAT16,
            DataType.BFLOAT16,
            DataType.INT8,
        ]

        if kernel_spec.data_type not in supported_dtypes:
            return False

        return True

    def profile(
        self, kernel_spec: KernelSpec, output_file: Optional[str] = None
    ) -> ProfileResult:
        """
        Profile the given kernel specification using ckProfiler.

        Args:
            kernel_spec: The kernel specification to profile
            output_file: Optional path to save the raw output (if not provided, uses a temp file)

        Returns:
            ProfileResult containing the profiling results

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
            start_time = datetime.now()

            print(f"Running ckProfiler with arguments: {' '.join(args)}")
            result = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
            )

            end_time = datetime.now()
            total_time_ms = (end_time - start_time).total_seconds() * 1000

            # Parse output file
            try:
                profile_data = self._parse_profiler_output(output_file)
            except Exception as e:
                raise RuntimeError(f"Failed to parse profiler output: {e}")

            # Create profile result
            return self._create_profile_result(kernel_spec, profile_data, total_time_ms)
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

    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the hardware used by this engine.

        Returns:
            Dictionary containing hardware information
        """
        if not self._initialized:
            self.initialize()

        return self._hardware_info

    def _find_profiler(self) -> Optional[str]:
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

    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect AMD GPU hardware information.

        Returns:
            Dictionary containing hardware information
        """
        hardware_info = {
            "vendor": "AMD",
            "device_type": "GPU",
            "timestamp": datetime.now().isoformat(),
        }

        # Try to get more detailed information using rocm-smi
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
                    if isinstance(rocm_info, dict) and "card" in rocm_info:
                        for card_id, card_info in rocm_info["card"].items():
                            # Just use the first card for now
                            hardware_info["device_name"] = card_info.get(
                                "card_model", "Unknown AMD GPU"
                            )
                            hardware_info["memory_size_mb"] = card_info.get(
                                "memory_usage", {}
                            ).get("total_memory", "Unknown")
                            hardware_info["clock_mhz"] = card_info.get(
                                "gfx_activity", {}
                            ).get("gfx_clock", "Unknown")
                            break
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Failed to parse rocm-smi output: {e}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Failed to run rocm-smi: {e}")

        # If we couldn't get detailed information, use generic values
        if "device_name" not in hardware_info:
            hardware_info["device_name"] = "AMD GPU"
            hardware_info["memory_size_mb"] = "Unknown"
            hardware_info["clock_mhz"] = "Unknown"

        return hardware_info

    def _build_profiler_args(
        self, kernel_spec: KernelSpec, params: GemmParams, output_file: str
    ) -> List[str]:
        """
        Build the arguments for the ckProfiler command.

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

        # Check the layout (assuming the specific format for ckProfiler)
        # Layout 0: A[m, k] * B[k, n] = C[m, n]
        layout = "0"

        # Other parameters
        verify = "0"  # Default to no verification for performance
        if self.config.verify_results:
            verify = "1"

        init = "1"  # Integer initialization
        print_tensor = "0"  # Don't print tensor values
        time_kernel = "1"  # Time the kernel
        warmup = str(self.config.warmup_iterations)
        iterations = str(kernel_spec.iterations)

        # Extract GEMM dimensions
        m = str(params.m)
        n = str(params.n)
        k = str(params.k)

        # Calculate strides (assuming row-major layout)
        stride_a = str(params.k)  # A[m,k] stride
        stride_b = str(params.n)  # B[k,n] stride
        stride_c = str(params.n)  # C[m,n] stride

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

    def _parse_profiler_output(self, output_file: str) -> Dict[str, Any]:
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

            # Parse the JSON data
            # ckProfiler outputs JSONL, but we'll just use the first line for now
            profile_data = json.loads(lines[0])

            return profile_data
        except (json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Failed to parse profiler output: {e}")

    def _create_profile_result(
        self,
        kernel_spec: KernelSpec,
        profile_data: Dict[str, Any],
        total_time_ms: float,
    ) -> ProfileResult:
        """
        Create a ProfileResult from the parsed profiler output.

        Args:
            kernel_spec: The kernel specification
            profile_data: Parsed profiler output
            total_time_ms: Total execution time in milliseconds

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

        if not kernel_times_ms:
            # If no kernel times were found, use a default value
            kernel_times_ms = [
                total_time_ms / kernel_spec.iterations
            ] * kernel_spec.iterations

        # Create timing data
        timing = TimingData(
            kernel_times_ms=kernel_times_ms,
            total_time_ms=total_time_ms,
            warmup_time_ms=None,  # We don't have this information
            num_iterations=len(kernel_times_ms),
            num_warmup=self.config.warmup_iterations,
        )

        # Extract metrics
        metrics = {}

        # Extract GFLOPS
        if "gflops" in profile_data:
            metrics["gflops"] = float(profile_data["gflops"])
        else:
            # Calculate GFLOPS based on the kernel specification
            gemm_params = kernel_spec.operation_params
            if isinstance(gemm_params, GemmOperationParams):
                params = gemm_params.params
                flops = (
                    2 * params.m * params.n * params.k
                )  # 2 operations per multiply-add
                avg_time_ms = timing.avg_kernel_time_ms
                if avg_time_ms > 0:
                    metrics["gflops"] = (flops / 1e9) / (avg_time_ms / 1000)

        # Create profile result
        return ProfileResult(
            kernel_spec=kernel_spec,
            timing=timing,
            metrics=metrics,
            engine_name="ComposableKernel",
            engine_info={"profiler_path": self._profiler_path},
            hardware_info=self._hardware_info,
            verification_result=None,  # We don't have this information
            raw_data=profile_data,
        )


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
                        print(f"  ✗ Failed for M={m}, N={n}, K={k}: {e}")

        print("")
        print("Scan completed!")

        # Save summary if output directory is specified
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return self.results
