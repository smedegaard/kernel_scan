"""
Profiling results module.

This module provides classes and functions for storing, analyzing, and
visualizing kernel profiling results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import polars as pl

from kernel_scan.core.logging import get_logger
from kernel_scan.core.specs import AcceleratorSpec, KernelSpec, OperationParams
from kernel_scan.core.units import BytesPerSecond, GigaFlops, Microsecond

log = get_logger(__name__)


@dataclass
class Metrics:
    """
    Container class for storing metrics for individual profiling runs.

    Attributes:
        latency: units.Microsecond. Time taken by the kernel to execute.
        memory_bandwidth: units.BytesPerSecond. Memory bandwidth achieved.
        compute_rate: units.GFLOPS. Compute rate achieved.
    """

    latency: Microsecond
    memory_bandwidth: BytesPerSecond
    compute_rate: GigaFlops

    def get(self, attr_name: str, default=None):
        """Get attribute value by name, returning default if not found."""
        return getattr(self, attr_name, default)


@dataclass
class ProfileResult:
    """
    Container for profiling results.

    Attributes:
        kernel_spec: The kernel specification that was profiled
        metrics: The measured values for the profile run.
        operation: Detailed operation description
        verification_result: Result of output verification (if performed)
        raw_data: Raw data from the profiling run
        is_best: Flag indicating if this is the best result in a comparison
    """

    kernel_spec: KernelSpec
    metrics: Metrics = field(default_factory=Metrics)
    operation: str = ""
    verification_result: Optional[bool] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile result to a dictionary for serialization."""
        # If raw_data is already populated with all the needed info, use it directly
        if self.raw_data and all(
            key in self.raw_data for key in ["operation", "time_ms", "tflops"]
        ):
            return self.raw_data.copy()

        # Otherwise, construct the dictionary from the ProfileResult attributes
        result_dict = {
            "operation": self.operation or f"{self.kernel_spec.name}",
            "is_best": self.is_best,
        }

        # Add metrics
        result_dict.update(self.metrics)

        # Extract operation parameters
        if self.kernel_spec.operation_type.name.lower() == "gemm":
            # Extract GEMM parameters
            params = self.kernel_spec.operation_params
            if hasattr(params, "__dict__"):
                params_dict = params.__dict__
                if "params" in params_dict and hasattr(
                    params_dict["params"], "__dict__"
                ):
                    gemm_params = params_dict["params"].__dict__
                    result_dict.update(
                        {
                            "M": gemm_params.get("m", 0),
                            "N": gemm_params.get("n", 0),
                            "K": gemm_params.get("k", 0),
                        }
                    )

                    # Add layout information if available
                    for layout_key in ["layout_a", "layout_b", "layout_c"]:
                        if layout_key in gemm_params and hasattr(
                            gemm_params[layout_key], "name"
                        ):
                            result_dict[layout_key] = gemm_params[
                                layout_key
                            ].name.replace("_", "")

        # Add data type information
        data_type = self.kernel_spec.data_type.name.lower()
        result_dict.update(
            {
                "datatype": data_type,
                "input_datatype": data_type,
                "weight_datatype": data_type,
                "output_datatype": data_type,
                "operation_type": self.kernel_spec.operation_type.name.lower(),
            }
        )

        # Add timestamp (use current time if not available)

        result_dict["timestamp"] = datetime.now().isoformat()

        return result_dict


class ProfileResultSet:
    """
    Container for multiple profiling results, with analysis capabilities.

    This class provides methods for storing, analyzing, and comparing multiple
    profiling results using Polars DataFrames.
    """

    def __init__(
        self,
        results: Optional[List[ProfileResult]] = None,
        accelerator_specs: AcceleratorSpec = AcceleratorSpec(),
        operation_params: OperationParams = OperationParams(),
    ):
        """
        Initialize a new ProfileResultSet.

        Args:
            results: Optional list of ProfileResult objects to initialize with
        """
        self._results = results or []
        self._df = None
        self._engine_name = "unknown"
        self._engine_info = {}
        self.accelerator_specs = accelerator_specs

    @property
    def engine_name(self) -> str:
        """Get the engine name used for profiling."""
        return self._engine_name

    @engine_name.setter
    def engine_name(self, name: str):
        """Set the engine name used for profiling."""
        self._engine_name = name

    @property
    def engine_info(self) -> Dict[str, Any]:
        """Get additional information about the engine."""
        return self._engine_info

    @engine_info.setter
    def engine_info(self, info: Dict[str, Any]):
        """Set additional information about the engine."""
        self._engine_info = info

    @property
    def hardware_info(self) -> Dict[str, Any]:
        """Get information about the hardware used."""
        return self._hardware_info

    @hardware_info.setter
    def hardware_info(self, info: Dict[str, Any]):
        """Set information about the hardware used."""
        self._hardware_info = info

    def add_result(self, result: ProfileResult) -> None:
        """
        Add a single profiling result to the set.

        Args:
            result: ProfileResult object to add
        """
        self._results.append(result)
        self._df = None  # Invalidate cached DataFrame

    def add_results(self, results: List[ProfileResult]) -> None:
        """
        Add multiple profiling results to the set.

        Args:
            results: List of ProfileResult objects to add
        """
        self._results.extend(results)
        self._df = None  # Invalidate cached DataFrame

    @property
    def results(self) -> List[ProfileResult]:
        """
        Get all profiling results.
        """
        return self._results

    @property
    def dataframe(self) -> pl.DataFrame:
        """
        Get a Polars DataFrame of all results.
        """
        data = [result.to_dict() for result in self._results]
        # Create DataFrame
        self._df = pl.DataFrame(data)

        return self._df

    def compare(
        self, group_by: str, metric: str = "time_ms"
    ) -> Union["pl.DataFrame", Dict[str, Any]]:
        """
        Compare results grouped by a specific attribute.

        Args:
            group_by: Attribute to group by (e.g., 'datatype', 'operation')
            metric: Metric to compare (e.g., 'time_ms', 'tflops')

        Returns:
            A DataFrame with grouped comparison results
        """
        df = self.dataframe
        if len(df) == 0:
            return pl.DataFrame()

        # Group by the specified attribute and calculate summary statistics
        return df.group_by(group_by).agg(
            [
                pl.mean(metric).alias(f"{metric}_mean"),
                pl.min(metric).alias(f"{metric}_min"),
                pl.max(metric).alias(f"{metric}_max"),
                pl.std(metric).alias(f"{metric}_std"),
                pl.count().alias("count"),
            ]
        )

    def mark_best_results(
        self, metric: str = "latency", lower_is_better: bool = True
    ) -> None:
        """
        Mark the best results in each group based on a metric.

        Args:
            metric: Metric to use for comparison (e.g., 'latency', 'compute_rate')
            lower_is_better: Whether lower values of the metric are better
        """
        if not self._results:
            return

        # Group results by operation type and data type
        grouped_results = {}
        for result in self._results:
            key = (result.kernel_spec.operation_type, result.kernel_spec.data_type)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)

        # Mark the best result in each group
        for group in grouped_results.values():
            if lower_is_better:
                best_result = min(
                    group, key=lambda r: r.metrics.get(metric, float("inf"))
                )
            else:
                best_result = max(group, key=lambda r: r.metrics.get(metric, 0.0))
            best_result.is_best = True
