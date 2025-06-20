"""
Plotting functions for kernel profiling results.

This module provides functions for visualizing kernel profiling results using Plotly.
"""

import logging
from typing import Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from kernel_scan.core.results import ProfileResultSet
from kernel_scan.core.types import DataType

log = logging.getLogger(__name__)


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


def calculate_gemm_roofline_data(
    result_set: ProfileResultSet,
    precision: DataType = DataType.FLOAT32,
    compute_unit: str = "tflops",
) -> "pl.DataFrame":
    """
    Calculate data needed for roofline model visualization.

    Filters for best results only

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
        log.warning("Empty dataframe in result_set")
        return pl.DataFrame()

    log.debug(f"Initial dataframe has {len(df)} rows")
    log.debug(f"Columns: {df.columns}")

    if "is_best" in df.columns:
        # Handle both boolean and string representations
        df = df.filter((pl.col("is_best") == True) | (pl.col("is_best") == "true"))
        log.info(f"Filtered to {len(df)} best results")
    else:
        log.warning("No 'is_best' column found - using all results")

    if len(df) == 0:
        log.warning("No best results found after filtering")
        return pl.DataFrame()

    # Get peak performance for the specified precision
    peak_compute = result_set.accelerator_specs.get_peak_compute(precision)
    peak_bandwidth = result_set.accelerator_specs.peak_bandwidth

    log.info(
        f"Peak compute: {peak_compute} TFLOPS, Peak bandwidth: {peak_bandwidth} GB/s"
    )

    # Ensure we have tflops column
    if "tflops" not in df.columns and "gflops" in df.columns:
        df = df.with_columns([(pl.col("gflops") / 1000).alias("tflops")])
    elif "tflops" not in df.columns:
        log.error("No tflops or gflops column found")
        return pl.DataFrame()

    df = df.with_columns(
        [
            (
                (pl.col("M") * pl.col("N") * pl.col("K"))
                / (
                    pl.col("M") * pl.col("K")
                    + pl.col("N") * pl.col("K")
                    + pl.col("M") * pl.col("N")
                )
            ).alias("gemm_arithmetic_intensity"),
            (
                pl.col("M").cast(pl.Utf8)
                + "_"
                + pl.col("N").cast(pl.Utf8)
                + "_"
                + pl.col("K").cast(pl.Utf8)
            ).alias("group"),
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

    if "time_ms" in df.columns:
        df = df.with_columns([(pl.col("time_ms").sqrt()).alias("time_scaled")])
    elif "avg_kernel_time_ms" in df.columns:
        df = df.with_columns(
            [
                pl.col("avg_kernel_time_ms").alias("time_ms"),
                (pl.col("avg_kernel_time_ms").sqrt()).alias("time_scaled"),
            ]
        )

    # Add precision information
    df = df.with_columns(
        [
            pl.lit(precision.name).alias("precision_format"),
            pl.lit(peak_compute).alias("peak_compute"),
            pl.lit(peak_bandwidth).alias("peak_bandwidth"),
        ]
    )

    log.debug(f"Final dataframe has {len(df)} rows")
    log.debug(
        f"Sample arithmetic intensity values: {df['gemm_arithmetic_intensity'].to_list()[:3]}"
    )
    log.debug(f"Sample tflops values: {df['tflops'].to_list()[:3]}")

    return df


def generate_gemm_roofline_plots_by_data_type(
    result_sets: List[ProfileResultSet],
    data_type: DataType,
    group_by: str = "M",
    title_prefix: Optional[str] = None,
) -> Dict[str, "go.Figure"]:
    """
    Generate interactive roofline plots for a specific data type.

    with growing N=K values.

    Args:
        result_sets: List of ProfileResultSet objects to visualize (one per M,N,K config)
        data_type: Data type to generate plots for
        group_by: Column name to group by within each data type plot (default is 'M')
        title_prefix: Optional prefix for plot titles

    Returns:
        Dictionary with a single entry keyed by data type name
    """

    # Combine all result sets for this data type
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
        # Calculate roofline data for this precision
        df = calculate_gemm_roofline_data(combined_result_set, data_type, "tflops")

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
        df = df.sort("gemm_arithmetic_intensity")

        title_base = f"{title_prefix} " if title_prefix else ""

        fig = px.scatter(
            df,
            x="gemm_arithmetic_intensity",
            y="tflops",
            log_x=True,
            log_y=True,
            color=group_by,
            color_discrete_sequence=px.colors.qualitative.G10,
            size="time_scaled" if "time_scaled" in df.columns else None,
            hover_data={
                "time_scaled": False,
                "time_ms": True,
                "M": True,
                "N": True,
                "K": True,
            }
            if all(
                col in df.columns for col in ["time_scaled", "time_ms", "M", "N", "K"]
            )
            else None,
            labels={
                "gemm_arithmetic_intensity": "Arithmetic Intensity (FLOPs/Byte)",
                "tflops": f"Performance (TFLOPs - {data_type.name})",
                "time_ms": "Execution Time (ms)",
            },
            title=f"{title_base}{hardware_name} Roofline Analysis - {data_type.name}",
            height=600,
        )

        # Create roofline line with more points for smoother curve
        x_min = float(df["gemm_arithmetic_intensity"].min())
        x_max = float(df["gemm_arithmetic_intensity"].max())

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


# Keep the original function for backward compatibility but update it to use the new approach
def generate_gemm_roofline_plots_by_group(
    result_set: ProfileResultSet,
    precision: DataType = DataType.FLOAT32,
    group_by: str = "M",
    title_prefix: Optional[str] = None,
) -> Dict[str, "go.Figure"]:
    """
    Generate interactive roofline plots grouped by a parameter.

    Updated to work with the new roofline calculation approach.

    Args:
        result_set: ProfileResultSet containing profiling results to visualize
        precision: Precision format to use for peak performance
        group_by: Column name to group by (default is 'M')
        title_prefix: Optional prefix for plot titles

    Returns:
        Dictionary of Plotly figures keyed by group values
    """

    # Use the new function with a single result set
    return generate_gemm_roofline_plots_by_data_type(
        result_sets=[result_set],
        data_type=precision,
        group_by=group_by,
        title_prefix=title_prefix,
    )
