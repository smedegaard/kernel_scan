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
from kernel_scan.core.types import DataType

log = get_logger(__name__)


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
    def _create_roofline_figure(
        cls,
        df: "pl.DataFrame",
        title: str,
        color_by: Optional[str] = None,
        size_column: str = "latency_value",
        hover_data: Optional[Dict[str, bool]] = None,
    ) -> "go.Figure":
        """
        Helper method to create a roofline plot figure from processed DataFrame.

        Args:
            df: DataFrame with roofline data
            title: Plot title
            color_by: Column to use for coloring data points
            size_column: Column to use for sizing data points
            hover_data: Additional data to display on hover

        Returns:
            Plotly Figure object
        """
        try:
            # Prepare scatter plot parameters
            scatter_params = {
                "x": "arithmetic_intensity_value",
                "y": "compute_performance_value",
                "log_x": True,
                "log_y": True,
                "size": size_column,
                "color_discrete_sequence": px.colors.qualitative.G10,
                "labels": {
                    "arithmetic_intensity_value": f"Arithmetic Intensity ({df['arithmetic_intensity_unit'].first()})",
                    "compute_performance_value": f"Compute Performance ({df['compute_performance_unit'].first()})",
                    "latency_value": f"Execution Time ({df['latency_unit'].first()})",
                },
                "title": title,
                "height": 600,
                "hover_data": {},
            }

            # Add color parameter if provided
            if color_by:
                # Cast the color column to string to ensure proper categorical coloring
                df = df.with_columns(
                    pl.col(color_by).cast(pl.String).alias(f"{color_by}_str")
                )
                scatter_params["color"] = f"{color_by}_str"
                scatter_params["hover_data"][f"{color_by}_str"] = False

            # Add hover data if provided
            if hover_data:
                scatter_params["hover_data"] = hover_data

            # Create scatter plot
            fig = px.scatter(df, **scatter_params)

            # Sort DataFrame by arithmetic intensity for proper line plotting
            sorted_df = df.sort("arithmetic_intensity_value")

            # Add peak compute line if available
            if "peak_compute_value" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sorted_df["arithmetic_intensity_value"].to_list(),
                        y=sorted_df["peak_compute_value"].to_list(),
                        mode="lines",
                        line=dict(color="blue", width=2),
                        opacity=0.1,
                        name=f"Peak Compute ({df['peak_compute_unit'].first()})",
                    )
                )

            if "memory_constraint_value" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sorted_df["arithmetic_intensity_value"].to_list(),
                        y=sorted_df["memory_constraint_value"].to_list(),
                        mode="lines",
                        line=dict(color="green", width=2),
                        opacity=0.1,
                        name=f"Peak Bandwidth ({df['memory_constraint_unit'].first()})",
                    )
                )

            # Add roofline line
            fig.add_trace(
                go.Scatter(
                    x=sorted_df["arithmetic_intensity_value"].to_list(),
                    y=sorted_df["attainable_performance_value"].to_list(),
                    mode="lines",
                    line=dict(color="tomato", width=2),
                    opacity=0.7,
                    name="Roofline",
                )
            )

            return fig

        except Exception:
            import traceback

            log.error(traceback.format_exc())
            return go.Figure()

    @classmethod
    def generate_roofline_plot(
        cls,
        result_set: ProfileResultSet,
        title_prefix: Optional[str] = None,
    ) -> "go.Figure":
        """
        Generate interactive roofline plots for a specific data type.

        Args:
            result_set: ProfileResultSet object to visualize
            title_prefix: Optional prefix for plot title

        Returns:
            Plotly Figure object
        """
        # Extract metadata
        kernel_spec = result_set.kernel_spec
        operation_params = kernel_spec.operation_params
        data_type = result_set.kernel_spec.data_type
        hardware_name = result_set.accelerator_spec.name

        # Calculate roofline data using operation-specific implementation
        df = cls.calculate_roofline_data(result_set)

        # Create title
        title_base = f"{title_prefix} " if title_prefix else ""
        title = (
            f"{title_base}{hardware_name} Roofline Analysis - {data_type.name} "
            f"({cls.get_operation_type().name})\n"
            f"m: {operation_params.m}, n: {operation_params.n},  k: {operation_params.k}"
        )

        # Create and return the figure using the helper method
        return cls._create_roofline_figure(df, title)

    @classmethod
    def generate_roofline_plots_by_group(
        cls, result_sets: List[ProfileResultSet], group_by: str
    ) -> Dict[str, go.Figure]:
        """
        Generate roofline plots grouped by a specific attribute.

        Args:
            result_sets: List of ProfileResultSet objects to visualize
            group_by: Column name to group by

        Returns:
            Dictionary of Plotly figures keyed by group value
        """
        figures = {}

        # Handle empty result sets
        if not result_sets:
            log.warning("No result sets provided for plotting")
            return figures

        # Group result sets by the grouping attribute first
        grouped_results = {}

        # Extract the group value from each result set and group them
        for rs in result_sets:
            try:
                # Calculate roofline data for this result
                df = cls.calculate_roofline_data(rs)

                # Extract the group value
                if group_by not in df.columns:
                    log.warning(
                        f"Group column '{group_by}' not found in result, skipping"
                    )
                    continue

                group_value = str(df[group_by].first())

                if group_value not in grouped_results:
                    grouped_results[group_value] = []

                # Store both the result set and its calculated dataframe
                grouped_results[group_value].append({"result_set": rs, "dataframe": df})
            except Exception as e:
                log.warning(f"Error processing result set: {e}")
                continue

        # Now process each group separately
        for group, group_data in grouped_results.items():
            if not group_data:
                continue

            # Process dataframes for this group
            dfs_with_ids = []
            metadata_map = {}

            for idx, data in enumerate(group_data):
                rs = data["result_set"]
                df = data["dataframe"]

                # Add a result_id column to track which result set this data came from
                df = df.with_columns(pl.lit(idx).alias("result_id"))
                dfs_with_ids.append(df)

                # Store metadata for this result set
                metadata_map[idx] = {
                    "kernel_spec": rs.kernel_spec,
                    "operation_params": rs.kernel_spec.operation_params,
                    "data_type": rs.kernel_spec.data_type,
                    "hardware_name": rs.accelerator_spec.name,
                }

            # Skip empty groups
            if not dfs_with_ids:
                continue

            try:
                # Concatenate only the dataframes for this specific group
                combined_df = pl.concat(dfs_with_ids)

                # Filter for best results if available
                if "is_best" in combined_df.columns:
                    combined_df = combined_df.filter(pl.col("is_best"))

                # Force categorical coloring of group_by
                combined_df = combined_df.sort(group_by)
                combined_df = combined_df.cast({group_by: pl.String})

                # Scale latency for better visualization
                combined_df = combined_df.with_columns(
                    (pl.col("latency_value").sqrt().alias("latency_scaled"))
                )

                # Get the first result_id for this group to retrieve metadata
                result_id = combined_df["result_id"].first()
                metadata = metadata_map[result_id]

                operation_params = metadata["operation_params"]
                data_type = metadata["data_type"]
                hardware_name = metadata["hardware_name"]

                # Create title with metadata information
                title = (
                    f"Roofline Analysis: {group_by} = {group}\n"
                    f"{hardware_name} - {data_type.name} ({cls.get_operation_type().name})\n"
                    f"m: {operation_params.m}, n: {operation_params.n},  k: {operation_params.k}"
                )

                # Create figure using the helper method
                fig = cls._create_roofline_figure(
                    combined_df,
                    title,
                    color_by="latency_scaled",
                    size_column="latency_scaled",
                    hover_data={"latency_value": True},
                )

                figures[group] = fig
            except Exception as e:
                log.warning(f"Error creating plot for group {group}: {e}")
                continue

        return figures
