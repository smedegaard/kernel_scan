"""
Profiling results module.

This module provides classes and functions for storing, analyzing, and
visualizing kernel profiling results.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import polars as pl

from kernel_scan.core.specs import KernelSpec


@dataclass
class TimingData:
    """
    Container for timing-related profiling results.

    Attributes:
        kernel_times_ms: List of kernel execution times in milliseconds
        total_time_ms: Total execution time including overhead in milliseconds
        warmup_time_ms: Time spent in warmup iterations in milliseconds
        overhead_time_ms: Time spent in overhead (setup, teardown, etc.) in milliseconds
        num_iterations: Number of timing iterations performed
        num_warmup: Number of warmup iterations performed
        timestamp: When the profiling was performed
    """

    kernel_times_ms: List[float]
    total_time_ms: float
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
        engine_name: Name of the engine used for profiling
        engine_info: Additional information about the engine
        hardware_info: Information about the hardware used
        verification_result: Result of output verification (if performed)
        raw_data: Raw data from the profiling run
    """

    kernel_spec: KernelSpec
    timing: TimingData
    metrics: Dict[str, float] = field(default_factory=dict)
    engine_name: str = "unknown"
    engine_info: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    verification_result: Optional[bool] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile result to a dictionary for serialization."""
        return {
            "operation_type": self.kernel_spec.operation_type.name,
            "data_type": self.kernel_spec.data_type.name,
            "kernel_name": self.kernel_spec.name or "unnamed",
            "engine": self.engine_name,
            "iterations": self.kernel_spec.iterations,
            "avg_time_ms": self.timing.avg_kernel_time_ms,
            "min_time_ms": self.timing.min_kernel_time_ms,
            "max_time_ms": self.timing.max_kernel_time_ms,
            "median_time_ms": self.timing.median_kernel_time_ms,
            "stddev_time_ms": self.timing.stddev_kernel_time_ms,
            "cv_percent": self.timing.cv_percent,
            **self.metrics,
            "verification_passed": self.verification_result,
            "timestamp": self.timing.timestamp.isoformat(),
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
        self.results = results or []
        self._df = None

    def add_result(self, result: ProfileResult) -> None:
        """
        Add a single profiling result to the set.

        Args:
            result: ProfileResult object to add
        """
        self.results.append(result)
        self._df = None  # Invalidate cached DataFrame

    def add_results(self, results: List[ProfileResult]) -> None:
        """
        Add multiple profiling results to the set.

        Args:
            results: List of ProfileResult objects to add
        """
        self.results.extend(results)
        self._df = None  # Invalidate cached DataFrame

    @property
    def dataframe(self) -> Union["pl.DataFrame", List[Dict[str, Any]]]:
        """
        Get a Polars DataFrame of all results.
        """
        if self._df is None or len(self._df) != len(self.results):
            if not self.results:
                return pl.DataFrame()

            # Convert all results to dictionaries
            data = [result.to_dict() for result in self.results]

            # Create DataFrame
            self._df = pl.DataFrame(data)

        return self._df

    def compare(
        self, group_by: str, metric: str = "avg_time_ms"
    ) -> Union["pl.DataFrame", Dict[str, Any]]:
        """
        Compare results grouped by a specific attribute.

        Args:
            group_by: Attribute to group by (e.g., 'data_type', 'engine')
            metric: Metric to compare (e.g., 'avg_time_ms', 'gflops')

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
