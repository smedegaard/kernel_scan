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
    combined = ProfileResultSet(accelerator_specs=result_sets[0].accelerator_specs)
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
    @abstractmethod
    def get_default_hover_data(cls) -> Dict[str, bool]:
        """
        Get the default hover data configuration for this operation type.

        Returns:
            Dictionary mapping column names to boolean visibility values
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_group_by(cls) -> str:
        """
        Get the default column to group by for this operation type.

        Returns:
            Column name to use for grouping
        """
        pass

    @classmethod
    def generate_roofline_plots_by_data_type(
        cls,
        result_sets: List[ProfileResultSet],
        data_type: DataType,
        group_by: Optional[str] = None,
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
        # Use operation-specific default if not provided
        if group_by is None:
            group_by = cls.get_default_group_by()

        # Combine all result sets
        combined_result_set = combine_result_sets(result_sets)

        if not combined_result_set.results:
            log.warning(f"No results to plot for {data_type.name}")
            return {}

        # Get hardware name for title
        hardware_name = (
            combined_result_set.accelerator_specs.name
            if combined_result_set.accelerator_specs
            else "Unknown Accelerator"
        )

        try:
            # Calculate roofline data using operation-specific implementation
            df = cls.calculate_roofline_data(combined_result_set, data_type, "tflops")

            if len(df) == 0:
                log.warning(f"No data available for precision {data_type.name}")
                return {}

            # Force categorical coloring of group_by column
            if group_by in df.columns:
                df = df.sort(group_by)
                df = df.with_columns([pl.col(group_by).cast(pl.Utf8)])
            else:
                log.warning(
                    f"Group by column '{group_by}' not found in DataFrame for {data_type.name}"
                )
                return {}

            # Sort by arithmetic intensity for better line drawing
            df = df.sort("arithmetic_intensity")

            title_base = f"{title_prefix} " if title_prefix else ""

            # Get operation-specific hover data
            hover_data = cls.get_default_hover_data()

            # Create scatter plot
            fig = px.scatter(
                df,
                x="arithmetic_intensity",
                y="tflops",
                log_x=True,
                log_y=True,
                color=group_by,
                color_discrete_sequence=px.colors.qualitative.G10,
                size="time_scaled" if "time_scaled" in df.columns else None,
                hover_data=hover_data
                if all(col in df.columns for col in hover_data)
                else None,
                labels={
                    "arithmetic_intensity": "Arithmetic Intensity (FLOPs/Byte)",
                    "tflops": f"Performance (TFLOPs - {data_type.name})",
                    "time_ms": "Execution Time (ms)",
                },
                title=f"{title_base}{hardware_name} Roofline Analysis - {data_type.name} ({cls.get_operation_type().name})",
                height=600,
            )

            # Create roofline line with more points for smoother curve
            x_min = float(df["arithmetic_intensity"].min())
            x_max = float(df["arithmetic_intensity"].max())

            # Generate more points for a smoother roofline
            import numpy as np

            x_range = np.logspace(np.log10(x_min * 0.5), np.log10(x_max * 2), 100)
            peak_compute = float(df["peak_compute"].first())
            peak_bandwidth = float(df["peak_bandwidth"].first())

            roofline_y = np.minimum(peak_compute, peak_bandwidth * x_range)

            # Add the roofline as a reference line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=roofline_y,
                    mode="lines",
                    line=dict(color="tomato", width=2),
                    opacity=0.7,
                    name="Roofline",
                )
            )

            log.info(f"Generated roofline plot for {data_type.name}")
            return {data_type.name: fig}

        except Exception as e:
            log.error(f"Error generating plot for {data_type.name}: {e}")
            import traceback

            log.error(traceback.format_exc())
            return {}
