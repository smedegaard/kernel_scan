"""
Plotting functions for kernel profiling results.

This module provides functions for visualizing kernel profiling results using Plotly.
"""

import logging
from typing import Dict, Optional

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.types import DataType

log = logging.getLogger(__name__)


def calculate_gemm_roofline_data(
    result_set: ProfileResultSet,
    precision: DataType = DataType.FLOAT32,
    compute_unit: str = "tflops",
) -> "pl.DataFrame":
    """
    Calculate data needed for roofline model visualization.

    Args:
        result_set: ProfileResultSet containing GEMM profiling results
        precision: Precision format to use for peak performance
        compute_unit: Unit of performance measure ('tflops' or 'gflops')

    Returns:
        DataFrame with additional columns for roofline analysis
    """
    # Extract DataFrame from ProfileResultSet
    df = result_set.dataframe

    if len(df) == 0:
        return pl.DataFrame()

    # Get peak performance for the specified precision
    log.info(
        f"Peak Compute Performance for {precision} on {result_set.accelerator_specs.name}:"
    )
    peak_compute = result_set.accelerator_specs.get_peak_compute(precision)
    peak_bandwidth = result_set.accelerator_specs.peak_bandwidth

    # Add required columns if not present
    if (
        compute_unit.lower() == "gflops"
        and "gflops" in df.columns
        and "tflops" not in df.columns
    ):
        df = df.with_columns([(pl.col("gflops") / 1000).alias("tflops")])
    elif (
        compute_unit.lower() == "tflops"
        and "tflops" in df.columns
        and "gflops" not in df.columns
    ):
        df = df.with_columns([(pl.col("tflops") * 1000).alias("gflops")])

    # Calculate GEMM operations and data movement
    df = df.with_columns(
        [
            # Calculate FLOPs for GEMM
            (pl.col("M") * pl.col("N") * pl.col("K") * 2).alias("flops"),
            # Calculate memory accesses in bytes (assuming each element is 4 bytes for float32)
            (
                (
                    pl.col("M") * pl.col("K")
                    + pl.col("N") * pl.col("K")
                    + pl.col("M") * pl.col("N")
                )
                * 4
            ).alias("bytes_accessed"),
            # Add group column for labeling
            (
                pl.col("M").cast(pl.Utf8)
                + "_"
                + pl.col("N").cast(pl.Utf8)
                + "_"
                + pl.col("K").cast(pl.Utf8)
            ).alias("group"),
        ]
    )

    # Calculate the original arithmetic intensity (for backward compatibility)
    df = df.with_columns(
        [
            (pl.col("flops") / pl.col("bytes_accessed")).alias(
                "gemm_arithmetic_intensity"
            )
        ]
    )

    # Calculate performance to bandwidth ratio (TFLOPS/GB/s)
    df = df.with_columns(
        [
            # Calculate actual bandwidth used (GB/s)
            (pl.col("bytes_accessed") / (pl.col("time_ms") * 1e-3) / 1e9).alias(
                "actual_bandwidth_gbps"
            ),
            # Calculate TFLOPS/GB/s ratio
            (
                pl.col("tflops")
                / (pl.col("bytes_accessed") / (pl.col("time_ms") * 1e-3) / 1e9)
            ).alias("tflops_per_gbps"),
        ]
    )

    # Calculate memory constraint line
    df = df.with_columns(
        [
            (peak_bandwidth * pl.col("gemm_arithmetic_intensity")).alias(
                "memory_constraint"
            )
        ]
    )

    # Calculate attainable performance (min of compute peak and memory constraint)
    df = df.with_columns(
        [
            pl.min_horizontal(
                peak_compute, peak_bandwidth * pl.col("gemm_arithmetic_intensity")
            ).alias("attainable_performance")
        ]
    )

    log.debug("Calculating attainable performance")
    log.debug(df["attainable_performance"])

    # For convenience, add time in ms if it's not already there
    if "avg_kernel_time_ms" in df.columns and "time_ms" not in df.columns:
        df = df.with_columns([pl.col("avg_kernel_time_ms").alias("time_ms")])

    # Add precision information
    df = df.with_columns(
        [
            pl.lit(precision.name).alias("precision_format"),
            pl.lit(peak_compute).alias("peak_compute"),
            pl.lit(peak_bandwidth).alias("peak_bandwidth"),
        ]
    )

    return df


def generate_gemm_roofline_plots_by_group(
    result_set: ProfileResultSet,
    precision: DataType = DataType.FLOAT32,
    group_by: str = "M",
    title_prefix: Optional[str] = None,
) -> Dict[str, "go.Figure"]:
    """
    Generate interactive roofline plots grouped by a parameter.

    Args:
        result_set: ProfileResultSet containing profiling results to visualize
        precision: Precision format to use for peak performance
        group_by: Column name to group by (default is 'M')
        title_prefix: Optional prefix for plot titles

    Returns:
        Dictionary of Plotly figures keyed by group values
    """

    # Extract DataFrame from ProfileResultSet
    df = calculate_gemm_roofline_data(result_set)

    log.debug(f"DataFrame columns: {df.columns}")
    log.debug(f"DataFrame sample: {df.head(2)}")

    df = calculate_gemm_roofline_data(result_set, precision, "tflops")
    if len(df) == 0:
        return {}

    # Add debug logging to inspect the dataframe
    log.debug(f"DataFrame columns: {df.columns}")
    log.debug(f"DataFrame sample: {df.head(2)}")

    _peak_compute = result_set.accelerator_specs.get_peak_compute(precision)

    # Force categorical coloring of group_by column
    if group_by in df.columns:
        df = df.sort(group_by)
        df = df.with_columns([pl.col(group_by).cast(pl.Utf8)])
    else:
        log.warning(f"Group by column '{group_by}' not found in DataFrame")
        return {}

    # Get hardware name for title
    hardware_name = (
        result_set.accelerator_specs.name
        if result_set.accelerator_specs
        else "Unknown Accelerator"
    )
    precision_name = precision.name if precision else "Unknown Precision"
    title_base = f"{title_prefix} " if title_prefix else ""

    # Create figures dictionary
    figures = {}
    groups = df.select(group_by).unique().to_series().to_list()

    for group in groups:
        group_df = df.filter(pl.col(group_by) == group)
        # Sort by the new metric for the plot
        group_df = group_df.sort("tflops_per_gbps")

        # Create individual figure for each group
        fig = px.scatter(
            group_df,  # Use the DataFrame directly instead of converting to dicts
            x="tflops_per_gbps",  # Use the new metric for x-axis
            y="tflops",
            log_x=True,
            log_y=True,
            color=group_by,
            color_discrete_sequence=px.colors.qualitative.G10,
            size="time_scaled" if "time_scaled" in group_df.columns else None,
            # hover_data={
            #     "time_scaled": False,
            #     "time_ms": True,
            #     "M": True,
            #     "N": True,
            #     "K": True,
            # },
            labels={
                "tflops_per_gbps": "Performance/Bandwidth Ratio (TFLOPS/GB/s)",  # New x-axis label
                "tflops": f"Performance (TFLOPs - {precision_name})",
                "time_ms": "Execution Time (ms)",
            },
            title=f"{title_base}{hardware_name} Roofline Analysis - {precision_name} - {group_by} = {group}",
            height=600,
        )

        # Modify the roofline to work with the new x-axis
        x_values = group_df["gemm_arithmetic_intensity"].to_list()
        y_values = group_df[
            "attainable_performance"
        ].to_list()  # Use actual performance values

        # Add the roofline as a reference
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color="tomato", width=2),
                opacity=0.7,
                name="Roofline",
            )
        )

        figures[str(group)] = fig

    return figures
