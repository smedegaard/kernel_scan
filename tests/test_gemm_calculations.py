"""
Tests for the GEMM operation calculation functions.

This module tests the functions that calculate FLOPs, bytes moved, and
arithmetic intensity for GEMM operations.
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to sys.path to import kernel_scan
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    from kernel_scan.api.operations.gemm import (
        GemmParams,
        calculate_arithmetic_intensity,
        calculate_bytes_moved,
        calculate_flops,
    )
    from kernel_scan.core.types import DataType, Layout
    from kernel_scan.core.units import Byte, Flops, FlopsPerByte
except ImportError as e:
    raise e


class TestCalculateFlops:
    """Test the calculate_flops function for GEMM operations."""

    def test_basic_calculation(self):
        """Test basic FLOP calculation for a simple GEMM operation."""
        # C = A * B (alpha=1.0, beta=0.0)
        params = GemmParams(
            m=10,
            n=20,
            k=30,
            alpha=1.0,
            beta=0.0,
        )

        flops = calculate_flops(params)

        # Each element in C requires k multiply-adds (2*k flops)
        # For m=10, n=20, k=30: 10*20*30*2 = 12000 flops
        assert isinstance(flops, Flops)
        assert flops.value == 12000

    def test_with_alpha_scaling(self):
        """Test FLOP calculation when alpha != 1.0."""
        # C = 2.0 * A * B (alpha=2.0, beta=0.0)
        params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=2.0,
            beta=0.0,
        )

        flops = calculate_flops(params)

        # Base multiply-adds: 10*10*10*2 = 2000 flops
        # Alpha scaling: 10*10 = 100 additional flops
        # Total: 2100 flops
        assert flops.value == 2100

    def test_with_beta_scaling(self):
        """Test FLOP calculation when beta != 0.0."""
        # C = A * B + 0.5 * C (alpha=1.0, beta=0.5)
        params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.0,
            beta=0.5,
        )

        flops = calculate_flops(params)

        # Base multiply-adds: 10*10*10*2 = 2000 flops
        # Beta scaling: 10*10 = 100 additional flops
        # Total: 2100 flops
        assert flops.value == 2100

    def test_with_both_scalings(self):
        """Test FLOP calculation when both alpha != 1.0 and beta != 0.0."""
        # C = 1.5 * A * B + 0.5 * C
        params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.5,
            beta=0.5,
        )

        flops = calculate_flops(params)

        # Base multiply-adds: 10*10*10*2 = 2000 flops
        # Alpha scaling: 10*10 = 100 additional flops
        # Beta scaling: 10*10 = 100 additional flops
        # Total: 2200 flops
        assert flops.value == 2200


class TestCalculateBytesMoved:
    """Test the calculate_bytes_moved function for GEMM operations."""

    def test_basic_calculation(self):
        """Test basic bytes moved calculation for GEMM."""
        # C = A * B (beta=0.0, no reading of C required)
        params = GemmParams(
            m=10,
            n=20,
            k=30,
            alpha=1.0,
            beta=0.0,
        )
        dtype_size = 4  # Assuming FP32 (4 bytes)

        bytes_moved = calculate_bytes_moved(params, dtype_size)

        # A: 10*30*4 = 1200 bytes (read)
        # B: 30*20*4 = 2400 bytes (read)
        # C: 10*20*4 = 800 bytes (write only, not read since beta=0.0)
        # Total: 1200 + 2400 + 800 = 4400 bytes
        assert isinstance(bytes_moved, Byte)
        assert bytes_moved.value == 4400

    def test_with_beta_nonzero(self):
        """Test bytes moved when beta != 0.0 (requires reading C)."""
        # C = A * B + 0.5 * C (beta=0.5, requires reading C)
        params = GemmParams(
            m=10,
            n=20,
            k=30,
            alpha=1.0,
            beta=0.5,
        )
        dtype_size = 4  # Assuming FP32 (4 bytes)

        bytes_moved = calculate_bytes_moved(params, dtype_size)

        # A: 10*30*4 = 1200 bytes (read)
        # B: 30*20*4 = 2400 bytes (read)
        # C: 10*20*4 = 800 bytes (read, since beta != 0.0)
        # C: 10*20*4 = 800 bytes (write)
        # Total: 1200 + 2400 + 800 + 800 = 5200 bytes
        assert bytes_moved.value == 5200

    def test_different_data_sizes(self):
        """Test bytes moved with different data sizes."""
        params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.0,
            beta=0.0,
        )

        # Test with FP16 (2 bytes)
        bytes_moved_fp16 = calculate_bytes_moved(params, 2)
        # A: 10*10*2 = 200 bytes (read)
        # B: 10*10*2 = 200 bytes (read)
        # C: 10*10*2 = 200 bytes (write)
        # Total: 200 + 200 + 200 = 600 bytes
        assert bytes_moved_fp16.value == 600

        # Test with FP64 (8 bytes)
        bytes_moved_fp64 = calculate_bytes_moved(params, 8)
        # A: 10*10*8 = 800 bytes (read)
        # B: 10*10*8 = 800 bytes (read)
        # C: 10*10*8 = 800 bytes (write)
        # Total: 800 + 800 + 800 = 2400 bytes
        assert bytes_moved_fp64.value == 2400


class TestCalculateArithmeticIntensity:
    """Test the calculate_arithmetic_intensity function for GEMM operations."""

    def test_basic_calculation(self):
        """Test basic arithmetic intensity calculation."""
        # C = A * B (beta=0.0)
        params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.0,
            beta=0.0,
        )

        intensity = calculate_arithmetic_intensity(params, DataType.FLOAT32)

        # Flops: 10*10*10*2 = 2000 flops
        # Bytes (FP32=4 bytes):
        #   A: 10*10*4 = 400 bytes (read)
        #   B: 10*10*4 = 400 bytes (read)
        #   C: 10*10*4 = 400 bytes (write only)
        #   Total: 1200 bytes
        # Intensity: 2000/1200 = 1.667 flops/byte
        assert isinstance(intensity, FlopsPerByte)
        # Use relative tolerance to account for floating point precision
        assert pytest.approx(intensity.value, rel=1e-3) == 2000 / 1200

    def test_with_beta_nonzero(self):
        """Test arithmetic intensity when beta != 0.0."""
        # C = A * B + 0.5 * C (requires reading C)
        params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.0,
            beta=0.5,
        )

        intensity = calculate_arithmetic_intensity(params, DataType.FLOAT32)

        # Flops: 10*10*10*2 + 10*10 = 2000 + 100 = 2100 flops
        # Bytes (FP32=4 bytes):
        #   A: 10*10*4 = 400 bytes (read)
        #   B: 10*10*4 = 400 bytes (read)
        #   C: 10*10*4 = 400 bytes (read)
        #   C: 10*10*4 = 400 bytes (write)
        #   Total: 1600 bytes
        # Intensity: 2100/1600 = 1.3125 flops/byte
        assert pytest.approx(intensity.value, rel=1e-3) == 2100 / 1600

    def test_with_different_data_types(self):
        """Test arithmetic intensity with different data types."""
        params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.0,
            beta=0.0,
        )

        # FP16 (2 bytes)
        intensity_fp16 = calculate_arithmetic_intensity(params, DataType.FLOAT16)
        # Flops: 10*10*10*2 = 2000 flops
        # Bytes (FP16=2 bytes): (10*10 + 10*10 + 10*10) * 2 = 600 bytes
        # Intensity: 2000/600 = 3.333... flops/byte
        assert pytest.approx(intensity_fp16.value, rel=1e-3) == 2000 / 600

        # FP64 (8 bytes)
        intensity_fp64 = calculate_arithmetic_intensity(params, DataType.FLOAT64)
        # Flops: 10*10*10*2 = 2000 flops
        # Bytes (FP64=8 bytes): (10*10 + 10*10 + 10*10) * 8 = 2400 bytes
        # Intensity: 2000/2400 = 0.833... flops/byte
        assert pytest.approx(intensity_fp64.value, rel=1e-3) == 2000 / 2400

    def test_different_matrix_dimensions(self):
        """Test arithmetic intensity with different matrix dimensions.

        Per the NVIDIA documentation, arithmetic intensity for GEMM operations
        is influenced by the matrix dimensions and their relationship. The formula
        for arithmetic intensity is:

        AI = (2*M*N*K) / (M*K + K*N + M*N) for beta=0

        For large matrices, this approaches k/3 when m and n are large.
        """
        # SCENARIO 1: Fixed k with varying m,n
        # When k is fixed, larger m and n values lead to better arithmetic intensity
        # as the operation becomes more compute-bound
        small_params = GemmParams(
            m=32,
            n=32,
            k=256,
            alpha=1.0,
            beta=0.0,
        )

        large_params = GemmParams(
            m=256,
            n=256,
            k=256,
            alpha=1.0,
            beta=0.0,
        )

        small_intensity = calculate_arithmetic_intensity(small_params, DataType.FLOAT32)
        large_intensity = calculate_arithmetic_intensity(large_params, DataType.FLOAT32)

        # When m and n increase with fixed k, arithmetic intensity increases
        assert large_intensity.value > small_intensity.value

        # SCENARIO 2: Fixed m,n with varying k
        # This tests how k affects arithmetic intensity when m and n are fixed
        small_k_params = GemmParams(
            m=128,
            n=128,
            k=32,
            alpha=1.0,
            beta=0.0,
        )

        large_k_params = GemmParams(
            m=128,
            n=128,
            k=512,
            alpha=1.0,
            beta=0.0,
        )

        small_k_intensity = calculate_arithmetic_intensity(
            small_k_params, DataType.FLOAT32
        )
        large_k_intensity = calculate_arithmetic_intensity(
            large_k_params, DataType.FLOAT32
        )

        # When k increases with fixed m and n, arithmetic intensity increases
        assert large_k_intensity.value > small_k_intensity.value

        # SCENARIO 3: Comparing ratios of dimensions
        # This tests different aspect ratios with the same total elements
        # Tall-and-skinny, fat-and-short, and cube matrices
        tall_skinny_params = GemmParams(
            m=512,
            n=32,
            k=128,
            alpha=1.0,
            beta=0.0,
        )

        fat_short_params = GemmParams(
            m=32,
            n=512,
            k=128,
            alpha=1.0,
            beta=0.0,
        )

        cube_params = GemmParams(
            m=128,
            n=128,
            k=128,
            alpha=1.0,
            beta=0.0,
        )

        tall_skinny_intensity = calculate_arithmetic_intensity(
            tall_skinny_params, DataType.FLOAT32
        )
        fat_short_intensity = calculate_arithmetic_intensity(
            fat_short_params, DataType.FLOAT32
        )
        cube_intensity = calculate_arithmetic_intensity(cube_params, DataType.FLOAT32)

        # For the same k value, square m×n shapes (cube) should have better intensity
        # than rectangular ones (tall-skinny or fat-short)
        assert cube_intensity.value > tall_skinny_intensity.value
        assert cube_intensity.value > fat_short_intensity.value


class TestEdgeCases:
    """Test edge cases for GEMM calculation functions."""

    def test_small_dimensions(self):
        """Test with small matrix dimensions."""
        params = GemmParams(
            m=1,
            n=1,
            k=1,
            alpha=1.0,
            beta=0.0,
        )

        flops = calculate_flops(params)
        bytes_moved = calculate_bytes_moved(params, 4)  # FP32
        intensity = calculate_arithmetic_intensity(params, DataType.FLOAT32)

        # Flops: 1*1*1*2 = 2 flops
        assert flops.value == 2

        # Bytes: (1*1 + 1*1 + 1*1) * 4 = 12 bytes
        assert bytes_moved.value == 12

        # Intensity: 2/12 = 0.166... flops/byte
        assert pytest.approx(intensity.value, rel=1e-3) == 2 / 12

    def test_large_dimensions(self):
        """Test with large matrix dimensions."""
        # Large matrices with power of 2 dimensions
        params = GemmParams(
            m=2**10,  # 1024
            n=2**10,  # 1024
            k=2**10,  # 1024
            alpha=1.0,
            beta=0.0,
        )

        flops = calculate_flops(params)
        bytes_moved = calculate_bytes_moved(params, 4)  # FP32
        intensity = calculate_arithmetic_intensity(params, DataType.FLOAT32)

        # Flops: 2 * 2^10 * 2^10 * 2^10 = 2 * 2^30 = 2^31 flops
        expected_flops = 2 * (2**10) * (2**10) * (2**10)
        assert flops.value == expected_flops

        # Bytes: Read A (m*k*4) + Read B (k*n*4) + Write C (m*n*4)
        # = (2^10*2^10 + 2^10*2^10 + 2^10*2^10) * 4 = 3 * 2^20 * 4 = 3 * 2^22
        expected_bytes = (
            params.m * params.k + params.k * params.n + params.m * params.n
        ) * 4
        assert bytes_moved.value == expected_bytes

        # Arithmetic intensity: flops/bytes = (2 * m * n * k) / ((m*k + k*n + m*n) * dtype_size)
        # For cube matrices where m=n=k, this simplifies to: (2 * k^3) / (3 * k^2 * dtype_size)
        # = (2 * k) / (3 * dtype_size)
        # With k=2^10 and dtype_size=4: (2 * 2^10) / (3 * 4) = 2^11 / 12 ≈ 170.67
        expected_intensity = (2 * params.k) / (3 * 4)

        assert pytest.approx(intensity.value, rel=0.01) == expected_intensity

    def test_arithmetic_intensity_formula(self):
        """Test that arithmetic intensity follows the expected formula for different matrix shapes."""
        # For GEMM operations with beta=0, the formula for arithmetic intensity is:
        # AI = (2 * m * n * k) / ((m*k + k*n + m*n) * dtype_size)

        # Test case 1: Cube matrix (m=n=k)
        cube_params = GemmParams(
            m=256,
            n=256,
            k=256,
            alpha=1.0,
            beta=0.0,
        )

        # Test case 2: Small k, large m,n (m=n>>k)
        wide_params = GemmParams(
            m=1024,
            n=1024,
            k=16,
            alpha=1.0,
            beta=0.0,
        )

        # Test case 3: Small m,n, large k (m,n<<k)
        deep_params = GemmParams(
            m=16,
            n=16,
            k=1024,
            alpha=1.0,
            beta=0.0,
        )

        # Calculate arithmetic intensities
        dtype_size = DataType.get_size_bytes(DataType.FLOAT32)
        cube_intensity = calculate_arithmetic_intensity(cube_params, DataType.FLOAT32)
        wide_intensity = calculate_arithmetic_intensity(wide_params, DataType.FLOAT32)
        deep_intensity = calculate_arithmetic_intensity(deep_params, DataType.FLOAT32)

        # Exact calculations for comparison:

        # Cube matrix (m=n=k):
        # Flops = 2 * m * n * k = 2 * 256^3
        # Bytes = (m*k + k*n + m*n) * dtype_size = (3 * 256^2) * 4
        # AI = Flops/Bytes = (2 * 256^3) / ((3 * 256^2) * 4) = (2 * 256) / (3 * 4) = 512 / 12 ≈ 42.67
        cube_flops = 2 * cube_params.m * cube_params.n * cube_params.k
        cube_bytes = (
            cube_params.m * cube_params.k
            + cube_params.k * cube_params.n
            + cube_params.m * cube_params.n
        ) * dtype_size
        expected_cube_intensity = cube_flops / cube_bytes
        assert pytest.approx(cube_intensity.value, rel=0.01) == expected_cube_intensity

        # Wide matrix (m=n>>k):
        # When m=n>>k, the m*n term dominates in the bytes calculation
        # Flops = 2 * m * n * k = 2 * 1024^2 * 16
        # Bytes = (m*k + k*n + m*n) * dtype_size ≈ (m*n) * dtype_size = 1024^2 * 4
        # AI = Flops/Bytes ≈ (2 * 1024^2 * 16) / (1024^2 * 4) = 2 * 16 / 4 = 8
        wide_flops = 2 * wide_params.m * wide_params.n * wide_params.k
        wide_bytes = (
            wide_params.m * wide_params.k
            + wide_params.k * wide_params.n
            + wide_params.m * wide_params.n
        ) * dtype_size
        expected_wide_intensity = wide_flops / wide_bytes
        assert pytest.approx(wide_intensity.value, rel=0.01) == expected_wide_intensity

        # Deep matrix (m=n<<k):
        # When m,n<<k, the m*k and n*k terms dominate in the bytes calculation
        # Flops = 2 * m * n * k = 2 * 16^2 * 1024
        # Bytes = (m*k + k*n + m*n) * dtype_size ≈ (m*k + n*k) * dtype_size = (16 + 16) * 1024 * 4
        # AI = Flops/Bytes ≈ (2 * 16^2 * 1024) / ((16 + 16) * 1024 * 4) = (2 * 16^2) / (32 * 4) = 512 / 128 = 4
        deep_flops = 2 * deep_params.m * deep_params.n * deep_params.k
        deep_bytes = (
            deep_params.m * deep_params.k
            + deep_params.k * deep_params.n
            + deep_params.m * deep_params.n
        ) * dtype_size
        expected_deep_intensity = deep_flops / deep_bytes
        assert pytest.approx(deep_intensity.value, rel=0.01) == expected_deep_intensity

    def test_layout_independence(self):
        """Test that calculations are independent of matrix layouts."""
        # All layouts should produce the same results for these functions
        row_major_params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.0,
            beta=0.0,
            layout_a=Layout.ROW_MAJOR,
            layout_b=Layout.ROW_MAJOR,
            layout_c=Layout.ROW_MAJOR,
        )

        col_major_params = GemmParams(
            m=10,
            n=10,
            k=10,
            alpha=1.0,
            beta=0.0,
            layout_a=Layout.COLUMN_MAJOR,
            layout_b=Layout.COLUMN_MAJOR,
            layout_c=Layout.COLUMN_MAJOR,
        )

        # The calculation functions should be layout-independent
        assert (
            calculate_flops(row_major_params).value
            == calculate_flops(col_major_params).value
        )
        assert (
            calculate_bytes_moved(row_major_params, 4).value
            == calculate_bytes_moved(col_major_params, 4).value
        )
        assert (
            calculate_arithmetic_intensity(row_major_params, DataType.FLOAT32).value
            == calculate_arithmetic_intensity(col_major_params, DataType.FLOAT32).value
        )
