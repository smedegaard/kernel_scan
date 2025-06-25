"""
Core plotting abstractions for kernel profiling results.

This module provides abstract base classes and utilities for visualizing
kernel profiling results using Plotly for different operation types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from kernel_scan.core.logging import get_logger
from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.specs import AcceleratorSpec
from kernel_scan.core.types import DataType, OperationType

log = get_logger(__name__)


def combine_result_sets(result_sets: List[ProfileResultSet]) -> ProfileResultSet:
    """
    Combine multiple ProfileResultSet objects into a single one.

    Args:
        result_sets: List of ProfileResultSet objects to combine

    Returns:
        Combined ProfileResultSet
    """
    if not result_sets:
        return ProfileResultSet()

    # Use the first result_set as the base
    # TODO: not happy with this implementation
    combined = ProfileResultSet(accelerator_spec=result_sets[0].accelerator_spec)
    combined.engine_name = result_sets[0].engine_name
    combined.engine_info = result_sets[0].engine_info

    # Add all results from all result sets
    for result_set in result_sets:
        combined.add_results(result_set.results)

    return combined


class OperationPlotter(ABC):
    """
    Abstract base class for operation-specific plotting functionality.

    This class defines the interface for plotting results for different
    operation types (GEMM, Conv, etc.).
    """

    @classmethod
    @abstractmethod
    def calculate_roofline_data(
        cls,
        result_set: ProfileResultSet,
        precision: DataType,
        compute_unit: str = "tflops",
    ) -> "pl.DataFrame":
        """
        Calculate data needed for roofline model visualization.

        Args:
            result_set: ProfileResultSet containing profiling results
            precision: Precision format to use for peak performance
            compute_unit: Unit of performance measure (e.g., 'tflops', 'gflops')

        Returns:
            DataFrame with additional columns for roofline analysis
        """
        pass

    @classmethod
    @abstractmethod
    def get_operation_type(cls) -> OperationType:
        """
        Get the operation type this plotter handles.

        Returns:
            OperationType enum value
        """
        pass

    @classmethod
    def generate_roofline_plots(
        cls,
        result_set: ProfileResultSet,
        accelerator_spec: AcceleratorSpec,
        title_prefix: Optional[str] = None,
    ) -> Dict[str, "go.Figure"]:
        """
        Generate interactive roofline plots for a specific data type.

        Args:
            result_sets: List of ProfileResultSet objects to visualize
            data_type: Data type to generate plots for
            group_by: Column name to group by (defaults to operation-specific value)
            title_prefix: Optional prefix for plot titles

        Returns:
            Dictionary with a single entry keyed by data type name
        """
        # Combine all result sets

        # Get hardware name for title
        hardware_name = accelerator_spec.name

        # Calculate roofline data using operation-specific implementation
        df = cls.calculate_roofline_data(result_set)

        title_base = f"{title_prefix} " if title_prefix else ""

        try:  # Create scatter plot
            fig = px.scatter(
                df,
                x="arithmetic_intensity",
                y="tflops",
                log_x=True,
                log_y=True,
                # color=group_by,
                color_discrete_sequence=px.colors.qualitative.G10,
                size="time_scaled" if "time_scaled" in df.columns else None,
                # hover_data=hover_data
                # if all(col in df.columns for col in hover_data)
                # else None,
                # labels={
                #     "arithmetic_intensity": "Arithmetic Intensity (FLOPs/Byte)",
                #     "tflops": f"Performance (TFLOPs - {kernel_spec.data_type.name})",
                #     "time_ms": "Execution Time (ms)",
                # },
                # title=f"{title_base}{hardware_name} Roofline Analysis - {data_type.name} ({cls.get_operation_type().name})",
                height=600,
            )

            # Create roofline line with more points for smoother curve
            # x_min = float(df["y"].min())
            # x_max = float(df["arithmetic_intensity"].max())

            # peak_compute = float(df["peak_compute"].first())
            # peak_bandwidth = float(df["peak_bandwidth"].first())

            # roofline_y = np.minimum(peak_compute, peak_bandwidth)

            # # Add the roofline as a reference line
            # fig.add_trace(
            #     go.Scatter(
            #         x=attainable_performance,
            #         y=roofline_y,
            #         mode="lines",
            #         line=dict(color="tomato", width=2),
            #         opacity=0.7,
            #         name="Roofline",
            #     )
            # )

        except Exception:
            # log.error(f"Error generating plot for {data_type.name}: {e}")
            import traceback

            log.error(traceback.format_exc())
            return {}
