"""
Profiling results module.

This module provides classes and functions for storing, analyzing, and
visualizing kernel profiling results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
        operation_params: Operation paramaters used to generate the profiling results
        verification_result: Result of output verification (if performed)
        raw_data: Raw data from the profiling run
        is_best: Flag indicating if this is the best result in a comparison
    """

    kernel_spec: KernelSpec
    metrics: Metrics = field(default_factory=Metrics)
    operation_params: OperationParams = field(default_factory=OperationParams)
    verification_result: Optional[bool] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ProfileResult to a flattened dictionary suitable for DataFrame creation.

        This method flattens nested objects into a single-level dictionary with
        appropriate key prefixes to avoid key collisions and to make the structure
        compatible with Polars' DataFrame creation.

        Returns:
            A flattened dictionary representation of the ProfileResult.
        """
        result_dict = {}

        # Add basic properties
        result_dict["is_best"] = self.is_best

        result_dict["data_type"] = self.kernel_spec.data_type.name

        result_dict["kernel_name"] = self.kernel_spec.name

        result_dict["operation_type"] = self.kernel_spec.operation_type.name

        # Extract metrics properties
        result_dict["latency_value"] = self.metrics.latency.value
        result_dict["latency_unit"] = self.metrics.latency.symbol

        result_dict["memory_bandwidth_value"] = self.metrics.memory_bandwidth.value
        result_dict["memory_bandwidth_unit"] = self.metrics.memory_bandwidth.symbol

        result_dict["compute_rate_value"] = self.metrics.compute_rate.value
        result_dict["comput_rate_unit"] = self.metrics.compute_rate.symbol

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
        accelerator_spec: Optional[AcceleratorSpec] = None,
        kernel_spec: Optional[KernelSpec] = None,
    ):
        """
        Initialize a new ProfileResultSet.

        Args:
            results: Optional list of ProfileResult objects to initialize with
            accelerator_spec: Optional accelerator specification
            kernel_spec: Optional kernel specification
        """
        self._results = results or []
        self._df = None
        self._engine_name = None
        self._kernel_spec = kernel_spec
        self._hardware_info = {}

        # If kernel_spec is None but we have results, try to get it from the first result
        if self._kernel_spec is None and self._results:
            self._kernel_spec = self._results[0].kernel_spec

    @property
    def kernel_spec(self) -> Optional[KernelSpec]:
        """
        Get the kernel specification.

        If no kernel_spec was explicitly set, try to get it from the first result.
        """
        if self._kernel_spec is not None:
            return self._kernel_spec
        elif self._results:
            return self._results[0].kernel_spec
        return None

    @kernel_spec.setter
    def kernel_spec(self, value: Optional[KernelSpec]):
        """Set the kernel specification."""
        self._kernel_spec = value

    @property
    def engine_name(self) -> str:
        """Get the engine name used for profiling."""
        return self._engine_name

    @engine_name.setter
    def engine_name(self, name: str):
        """Set the engine name used for profiling."""
        self._engine_name = name

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
    def results_as_dataframe(self) -> pl.DataFrame:
        """
        Get a Polars DataFrame of all results.
        """
        data = [result.to_dict() for result in self._results]
        # Create DataFrame
        self._df = pl.DataFrame(data)

        return self._df

    # def compare(
    #     self, group_by: str, metric: str = "time_ms"
    # ) -> Union["pl.DataFrame", Dict[str, Any]]:
    #     """
    #     Compare results grouped by a specific attribute.

    #     Args:
    #         group_by: Attribute to group by (e.g., 'datatype', 'operation')
    #         metric: Metric to compare (e.g., 'time_ms', 'tflops')

    #     Returns:
    #         A DataFrame with grouped comparison results
    #     """
    #     df = self.dataframe
    #     if len(df) == 0:
    #         return pl.DataFrame()

    #     # Group by the specified attribute and calculate summary statistics
    #     return df.group_by(group_by).agg(
    #         [
    #             pl.mean(metric).alias(f"{metric}_mean"),
    #             pl.min(metric).alias(f"{metric}_min"),
    #             pl.max(metric).alias(f"{metric}_max"),
    #             pl.std(metric).alias(f"{metric}_std"),
    #             pl.count().alias("count"),
    #         ]
    #     )

    # TODO: change to take units.Dimension instead of str for `metric`
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
