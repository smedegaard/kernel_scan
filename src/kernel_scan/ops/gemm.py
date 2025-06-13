"""
GEMM operations implementation.

This module implements the General Matrix Multiplication (GEMM) operations
for the kernel_scan library. It provides the necessary functionality to
define and validate GEMM operations for GPU profiling.
"""

import math
import random
from typing import Optional, Tuple

# Import Polars - required for this module
import polars as pl

from kernel_scan.core.types import (
    DataType,
    GemmInputs,
    GemmOutputs,
    GemmParams,
    IncompatibleDataTypesError,
    IncompatibleDimensionsError,
    IncompatibleLayoutError,
    InvalidTensorShapeError,
    Layout,
)


def validate_gemm_operation(
    params: GemmParams, inputs: GemmInputs, outputs: GemmOutputs
) -> bool:
    """
    Validate that GEMM operation parameters, inputs, and outputs are consistent.

    Args:
        params: The GEMM operation parameters
        inputs: The input tensors (A and B)
        outputs: The output tensor (C)

    Returns:
        True if the operation is valid

    Raises:
        InvalidTensorShapeError: If any tensor shape is invalid
        IncompatibleDimensionsError: If tensor dimensions don't match the operation
        IncompatibleLayoutError: If tensor layouts don't match the operation parameters
        IncompatibleDataTypesError: If tensor data types are inconsistent
    """
    # Validate input dimensions
    if len(inputs.a.dimensions) != 2:
        raise InvalidTensorShapeError(
            f"Matrix A must be 2D, got {len(inputs.a.dimensions)}D"
        )
    if len(inputs.b.dimensions) != 2:
        raise InvalidTensorShapeError(
            f"Matrix B must be 2D, got {len(inputs.b.dimensions)}D"
        )
    if len(outputs.c.dimensions) != 2:
        raise InvalidTensorShapeError(
            f"Matrix C must be 2D, got {len(outputs.c.dimensions)}D"
        )

    # Extract dimensions based on layout
    if inputs.a.layout == Layout.ROW_MAJOR:
        a_rows, a_cols = inputs.a.dimensions
    else:
        a_cols, a_rows = inputs.a.dimensions

    if inputs.b.layout == Layout.ROW_MAJOR:
        b_rows, b_cols = inputs.b.dimensions
    else:
        b_cols, b_rows = inputs.b.dimensions

    if outputs.c.layout == Layout.ROW_MAJOR:
        c_rows, c_cols = outputs.c.dimensions
    else:
        c_cols, c_rows = outputs.c.dimensions

    # Validate dimensions match the GEMM operation
    if a_rows != params.m:
        raise IncompatibleDimensionsError(
            f"Matrix A rows ({a_rows}) doesn't match GEMM m parameter ({params.m})"
        )

    if a_cols != params.k:
        raise IncompatibleDimensionsError(
            f"Matrix A columns ({a_cols}) doesn't match GEMM k parameter ({params.k})"
        )

    if b_rows != params.k:
        raise IncompatibleDimensionsError(
            f"Matrix B rows ({b_rows}) doesn't match GEMM k parameter ({params.k})"
        )

    if b_cols != params.n:
        raise IncompatibleDimensionsError(
            f"Matrix B columns ({b_cols}) doesn't match GEMM n parameter ({params.n})"
        )

    if c_rows != params.m:
        raise IncompatibleDimensionsError(
            f"Matrix C rows ({c_rows}) doesn't match GEMM m parameter ({params.m})"
        )

    if c_cols != params.n:
        raise IncompatibleDimensionsError(
            f"Matrix C columns ({c_cols}) doesn't match GEMM n parameter ({params.n})"
        )

    # Validate layouts
    if inputs.a.layout != params.layout_a:
        raise IncompatibleLayoutError(
            f"Matrix A layout ({inputs.a.layout}) doesn't match specified layout ({params.layout_a})"
        )

    if inputs.b.layout != params.layout_b:
        raise IncompatibleLayoutError(
            f"Matrix B layout ({inputs.b.layout}) doesn't match specified layout ({params.layout_b})"
        )

    if outputs.c.layout != params.layout_c:
        raise IncompatibleLayoutError(
            f"Matrix C layout ({outputs.c.layout}) doesn't match specified layout ({params.layout_c})"
        )

    # Validate data types consistency
    if inputs.a.data_type != inputs.b.data_type:
        raise IncompatibleDataTypesError(
            f"Matrix A data type ({inputs.a.data_type}) doesn't match Matrix B data type ({inputs.b.data_type})"
        )

    if inputs.a.data_type != outputs.c.data_type:
        raise IncompatibleDataTypesError(
            f"Input matrices data type ({inputs.a.data_type}) doesn't match output matrix data type ({outputs.c.data_type})"
        )

    return True


def calculate_gemm_flops(params: GemmParams) -> int:
    """
    Calculate the number of floating-point operations for a GEMM operation.

    For matrix multiplication C = alpha * A * B + beta * C:
    - Each element in C requires k multiply-adds (2 operations per multiply-add)
    - If alpha != 1.0, add one multiplication per element in C
    - If beta != 0.0, add one multiplication per element in C

    Args:
        params: The GEMM operation parameters

    Returns:
        The number of floating-point operations
    """
    # Basic FLOPs for matrix multiplication: 2 * m * n * k
    # (k multiply-adds per element, 2 operations per multiply-add)
    flops = 2 * params.m * params.n * params.k

    # Add FLOPs for alpha scaling if alpha != 1.0
    if params.alpha != 1.0:
        flops += params.m * params.n

    # Add FLOPs for beta scaling if beta != 0.0
    if params.beta != 0.0:
        flops += params.m * params.n

    return flops


def create_random_gemm_inputs(
    params: GemmParams, data_type: DataType, seed: Optional[int] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create random input matrices for GEMM operation using Polars.

    Args:
        params: The GEMM operation parameters
        data_type: The data type for the matrices
        seed: Optional random seed for reproducibility

    Returns:
        Tuple of (A, B, C) matrices as Polars DataFrames
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Get polars dtype
    pl_dtype = DataType.get_polars_dtype(data_type)

    # Helper function to create random DataFrame
    def create_random_df(rows, cols, dtype=pl_dtype):
        return pl.DataFrame(
            {f"col_{i}": [random.random() for _ in range(rows)] for i in range(cols)}
        ).cast({f"col_{i}": dtype for i in range(cols)})

    # Create random matrices with Polars
    if params.layout_a == Layout.ROW_MAJOR:
        a = create_random_df(params.m, params.k)
    else:
        # For column major layout
        a = create_random_df(params.k, params.m)
        # Handle column major through metadata instead of transposing

    if params.layout_b == Layout.ROW_MAJOR:
        b = create_random_df(params.k, params.n)
    else:
        # For column major layout
        b = create_random_df(params.n, params.k)
        # Handle column major through metadata instead of transposing

    if params.layout_c == Layout.ROW_MAJOR:
        c = create_random_df(params.m, params.n)
    else:
        # For column major layout
        c = create_random_df(params.n, params.m)
        # Handle column major through metadata instead of transposing

    return a, b, c


def verify_gemm_result(
    a: pl.DataFrame,
    b: pl.DataFrame,
    c: pl.DataFrame,
    c_result: pl.DataFrame,
    params: GemmParams,
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify that a GEMM operation result is correct using Polars.

    Args:
        a: Input matrix A (Polars DataFrame)
        b: Input matrix B (Polars DataFrame)
        c: Input matrix C before the operation (Polars DataFrame)
        c_result: Result matrix C after the operation (Polars DataFrame)
        params: The GEMM operation parameters
        tolerance: Tolerance for floating point comparison

    Returns:
        True if the result is correct within the tolerance
    """
    # Implement matrix multiplication with Polars
    # This is a custom implementation since Polars doesn't have built-in matrix multiplication

    # Extract dimensions based on layout
    if params.layout_a == Layout.ROW_MAJOR:
        a_rows, a_cols = params.m, params.k
    else:
        a_rows, a_cols = params.k, params.m

    if params.layout_b == Layout.ROW_MAJOR:
        b_rows, b_cols = params.k, params.n
    else:
        b_rows, b_cols = params.n, params.k

    if params.layout_c == Layout.ROW_MAJOR:
        c_rows, c_cols = params.m, params.n
    else:
        c_rows, c_cols = params.n, params.m

    # Create an empty result matrix
    expected = pl.DataFrame()

    # Handle matrix multiplication
    # For each element in the result matrix, calculate the dot product
    for i in range(params.m):
        row_values = []
        for j in range(params.n):
            # Calculate dot product for this element
            element_sum = 0.0

            for k in range(params.k):
                # Get values based on layout
                if params.layout_a == Layout.ROW_MAJOR:
                    a_val = a.row(i)[k]
                else:
                    a_val = a.row(k)[i]

                if params.layout_b == Layout.ROW_MAJOR:
                    b_val = b.row(k)[j]
                else:
                    b_val = b.row(j)[k]

                element_sum += a_val * b_val

            # Apply alpha and beta
            if params.layout_c == Layout.ROW_MAJOR:
                c_val = c.row(i)[j]
            else:
                c_val = c.row(j)[i]

            result = params.alpha * element_sum + params.beta * c_val
            row_values.append(result)

        # Add the row to the expected result
        if not expected.is_empty():
            expected = expected.vstack(pl.DataFrame([row_values]))
        else:
            expected = pl.DataFrame([row_values])

    # Compare expected with actual result
    all_close = True

    for i in range(params.m):
        for j in range(params.n):
            expected_val = expected.row(i)[j]

            if params.layout_c == Layout.ROW_MAJOR:
                actual_val = c_result.row(i)[j]
            else:
                actual_val = c_result.row(j)[i]

            # Check if values are close within tolerance
            if not math.isclose(
                expected_val, actual_val, rel_tol=tolerance, abs_tol=tolerance
            ):
                all_close = False
                break
        if not all_close:
            break

    return all_close
