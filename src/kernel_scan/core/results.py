"""
Profiling results module.

This module provides classes and functions for storing, analyzing, and
visualizing kernel profiling results.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import polars as pl

from kernel_scan.core.specs import KernelSpec

log = logging.getLogger(__name__)


@dataclass
class TimingData:
    """
    Container for timing-related profiling results.

    Attributes:
        kernel_times_ms: List of kernel execution times in milliseconds
        warmup_time_ms: Time spent in warmup iterations in milliseconds
        overhead_time_ms: Time spent in overhead (setup, teardown, etc.) in milliseconds
        num_iterations: Number of timing iterations performed
        num_warmup: Number of warmup iterations performed
        timestamp: When the profiling was performed
    """

    kernel_times_ms: List[float]
    warmup_time_ms: Optional[float] = None
    overhead_time_ms: Optional[float] = None
    num_iterations: int = 0
    num_warmup: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def avg_kernel_time_ms(self) -> float:
        """Average kernel execution time in milliseconds."""
        if not self.kernel_times_ms:
            return 0.0
        return statistics.mean(self.kernel_times_ms)

    @property
    def min_kernel_time_ms(self) -> float:
        """Minimum kernel execution time in milliseconds."""
        if not self.kernel_times_ms:
            return 0.0
        return min(self.kernel_times_ms)

    @property
    def max_kernel_time_ms(self) -> float:
        """Maximum kernel execution time in milliseconds."""
        if not self.kernel_times_ms:
            return 0.0
        return max(self.kernel_times_ms)

    @property
    def median_kernel_time_ms(self) -> float:
        """Median kernel execution time in milliseconds."""
        if not self.kernel_times_ms:
            return 0.0
        return statistics.median(self.kernel_times_ms)

    @property
    def stddev_kernel_time_ms(self) -> float:
        """Standard deviation of kernel execution times in milliseconds."""
        if len(self.kernel_times_ms) < 2:
            return 0.0
        return statistics.stdev(self.kernel_times_ms)

    @property
    def cv_percent(self) -> float:
        """Coefficient of variation as a percentage."""
        if self.avg_kernel_time_ms == 0:
            return 0.0
        return (self.stddev_kernel_time_ms / self.avg_kernel_time_ms) * 100


@dataclass
class ProfileResult:
    """
    Container for profiling results.

    Attributes:
        kernel_spec: The kernel specification that was profiled
        timing: Timing data from the profiling run
        metrics: Additional metrics (e.g., GFLOPS, bandwidth)
        operation: Detailed operation description
        verification_result: Result of output verification (if performed)
        raw_data: Raw data from the profiling run
        is_best: Flag indicating if this is the best result in a comparison
    """

    kernel_spec: KernelSpec
    timing: TimingData
    metrics: Dict[str, float] = field(default_factory=dict)
    operation: str = ""
    verification_result: Optional[bool] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile result to a dictionary for serialization."""
        # Extract operation parameters
        op_params = {}
        if self.kernel_spec.operation_type.name.lower() == "gemm":
            # Extract GEMM parameters
            params = self.kernel_spec.operation_params.__dict__.get("params", {})
            if hasattr(params, "__dict__"):
                params_dict = params.__dict__
                op_params = {
                    "M": params_dict.get("m", 0),
                    "N": params_dict.get("n", 0),
                    "K": params_dict.get("k", 0),
                    "layout_a": params_dict.get("layout_a", "RowMajor").name.replace(
                        "_", ""
                    ),
                    "layout_b": params_dict.get("layout_b", "RowMajor").name.replace(
                        "_", ""
                    ),
                    "layout_c": params_dict.get("layout_c", "RowMajor").name.replace(
                        "_", ""
                    ),
                }

        # Extract data types
        data_type = self.kernel_spec.data_type.name.lower()

        return {
            "operation": self.operation or f"{self.kernel_spec.name}",
            "time_ms": self.timing.avg_kernel_time_ms,
            "tflops": self.metrics.get("tflops", 0.0),
            "gb_per_sec": self.metrics.get("bandwidth", 0.0),
            "is_best": self.is_best,
            "timestamp": self.timing.timestamp.isoformat(),
            "datatype": data_type,
            "input_datatype": data_type,
            "weight_datatype": data_type,
            "output_datatype": data_type,
            "operation_type": self.kernel_spec.operation_type.name.lower(),
            **op_params,
        }


class ProfileResultSet:
    """
    Container for multiple profiling results, with analysis capabilities.

    This class provides methods for storing, analyzing, and comparing multiple
    profiling results using Polars DataFrames.
    """

    def __init__(self, results: Optional[List[ProfileResult]] = None):
        """
        Initialize a new ProfileResultSet.

        Args:
            results: Optional list of ProfileResult objects to initialize with
        """
        self._results = results or []
        self._df = None
        self._engine_name = "unknown"
        self._engine_info = {}
        self._hardware_info = {}

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
    def dataframe(self) -> Union["pl.DataFrame", List[Dict[str, Any]]]:
        """
        Get a Polars DataFrame of all results.
        """
        if self._df is None or len(self._df) != len(self._results):
            if not self._results:
                return pl.DataFrame()

            # Convert all results to dictionaries
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
        self, metric: str = "time_ms", lower_is_better: bool = True
    ) -> None:
        """
        Mark the best results in each group based on a metric.

        Args:
            metric: Metric to use for comparison (e.g., 'time_ms', 'tflops')
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
