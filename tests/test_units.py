"""
Tests for the units module.

This module tests the functionality of the custom units system used
for GPU performance profiling.
"""

import pytest

from kernel_scan.core.units import (
    Bit,
    Byte,
    Flops,
    FlopsPerByte,
    GigaByte,
    GigaBytesPerSecond,
    GigaFlops,
    KiloByte,
    MegaByte,
    MegaFlops,
    Millisecond,
    Prefix,
    Second,
    TeraFlops,
    compute_arithmetic_intensity,
)


class TestUnitBasics:
    """Test basic unit functionality."""

    def test_unit_creation(self):
        """Test creating units with different prefixes."""
        # Base unit with no prefix
        second = Second(1.0)
        assert second.value == 1.0
        assert second.prefix == Prefix.NONE
        assert second.base_value == 1.0

        # Unit with prefix via class constructor
        millisecond = Millisecond(500)
        assert millisecond.value == 500
        assert millisecond.prefix == Prefix.MILLI
        assert millisecond.base_value == 0.5  # 500 ms = 0.5 s

        # Unit with explicit prefix
        giga_flops = Flops(2.5, Prefix.GIGA)
        assert giga_flops.value == 2.5
        assert giga_flops.prefix == Prefix.GIGA
        assert giga_flops.base_value == 2.5e9  # 2.5 GFLOPs = 2.5e9 FLOPS

    def test_unit_properties(self):
        """Test unit name and symbol generation."""
        flops = Flops(1.0)
        assert flops.name == "flops"
        assert flops.symbol == "FLOPS"

        gflops = GigaFlops(1.0)
        assert gflops.name == "Gigaflops"
        assert gflops.symbol == "GFLOPS"

        tflops = TeraFlops(1.0)
        assert tflops.name == "Teraflops"
        assert tflops.symbol == "TFLOPS"

    def test_string_representation(self):
        """Test string and representation methods."""
        second = Second(1.5)
        assert str(second) == "1.5 s"
        assert repr(second) == "Second(1.5, Prefix.NONE)"

        gflops = GigaFlops(2.5)
        assert str(gflops) == "2.5 GFLOPS"
        assert repr(gflops) == "GigaFlops(2.5, Prefix.GIGA)"

    def test_format_method(self):
        """Test format method with different precisions."""
        tflops = TeraFlops(1.23456)
        assert tflops.format() == "1.23 TFLOPS"  # default precision=2
        assert tflops.format(precision=1) == "1.2 TFLOPS"
        assert tflops.format(precision=4) == "1.2346 TFLOPS"


class TestUnitConversion:
    """Test unit conversion between prefixes and types."""

    def test_prefix_conversion(self):
        """Test conversion between different prefixes of the same unit."""
        # Start with teraflops and convert down
        tflops = TeraFlops(1.5)
        gflops = tflops.to_giga()
        assert isinstance(gflops, GigaFlops)
        assert gflops.value == 1500.0  # 1.5 TFLOPS = 1500 GFLOPS

        # Convert from base to various prefixes
        flops = Flops(1e12)  # 1 trillion FLOPS
        assert flops.to_tera().value == 1.0
        assert flops.to_giga().value == 1000.0
        assert flops.to_mega().value == 1000000.0

        # Using the with_prefix method
        kb = Byte(2048).with_prefix(Prefix.KILO)
        assert kb.value == 2.048

    def test_type_conversion(self):
        """Test conversion between different unit types."""
        # Bytes to bits
        byte = Byte(1.0)
        bit = byte.to(Bit)
        assert bit.value == 8.0  # 1 byte = 8 bits

        # Bits to bytes
        bits = Bit(16.0)
        bytes_val = bits.to(Byte)
        assert bytes_val.value == 2.0  # 16 bits = 2 bytes

        # Test conversions with prefixes
        kb = KiloByte(1.0)
        bits = kb.to(Bit)
        assert bits.value == 8000.0  # 1 KB = 8000 bits

    def test_incompatible_conversion(self):
        """Test that conversion between incompatible units raises ValueError."""
        flops = Flops(1.0)
        with pytest.raises(ValueError):
            flops.to(Byte)  # Can't convert FLOPS to Bytes

        byte = Byte(1.0)
        with pytest.raises(ValueError):
            byte.to(Second)  # Can't convert Bytes to Seconds


class TestMathOperations:
    """Test mathematical operations on units."""

    def test_addition(self):
        """Test adding units together."""
        # Adding same units
        a = Second(1.0)
        b = Second(2.0)
        c = a + b
        assert c.value == 3.0
        assert isinstance(c, Second)

        # Adding units with different prefixes
        a = GigaFlops(1.0)  # 1 GFLOPS
        b = MegaFlops(500.0)  # 500 MFLOPS = 0.5 GFLOPS
        c = a + b
        assert c.value == 1.5
        assert isinstance(c, GigaFlops)

        # Adding scalar
        a = TeraFlops(2.0)
        b = a + 1.0
        assert b.value == 3.0
        assert isinstance(b, TeraFlops)

    def test_subtraction(self):
        """Test subtracting units."""
        a = Second(5.0)
        b = Second(2.0)
        c = a - b
        assert c.value == 3.0

        # Subtracting with different prefixes
        a = GigaByte(2.0)
        b = MegaByte(500.0)  # 0.5 GB
        c = a - b
        assert c.value == 1.5
        assert isinstance(c, GigaByte)

        # Subtracting scalar
        a = TeraFlops(5.0)
        b = a - 2.0
        assert b.value == 3.0

    def test_multiplication(self):
        """Test multiplying units by scalars."""
        a = Second(2.0)
        b = a * 3
        assert b.value == 6.0
        assert isinstance(b, Second)

        # Test reverse multiplication
        c = 4 * a
        assert c.value == 8.0

    def test_division(self):
        """Test dividing units."""
        # Division by scalar
        a = GigaFlops(6.0)
        b = a / 2
        assert b.value == 3.0
        assert isinstance(b, GigaFlops)

        # Division by compatible unit (special case testing)
        compute = TeraFlops(1.2)
        bandwidth = GigaBytesPerSecond(0.3)
        intensity = compute_arithmetic_intensity(compute, bandwidth)
        assert isinstance(intensity, FlopsPerByte)
        assert intensity.value == 4.0  # 1.2 TFLOPS / 0.3 GB/s = 4 FLOPS/B


class TestComparison:
    """Test comparison operations."""

    def test_equality(self):
        """Test equality comparison."""
        a = Second(1.0)
        b = Second(1.0)
        c = Second(2.0)
        assert a == b
        assert a != c

        # Different prefixes but same base value
        a = GigaFlops(1.0)
        b = MegaFlops(1000.0)  # 1000 MFLOPs = 1 GFLOPS
        assert a == b

    def test_ordering(self):
        """Test ordering comparisons."""
        a = Second(1.0)
        b = Second(2.0)
        assert a < b
        assert b > a
        assert a <= b
        assert b >= a

        # Different prefixes
        a = KiloByte(1.0)  # 1 KB
        b = Byte(1500)  # 1.5 KB
        assert a < b
        assert b > a

    def test_incompatible_comparison(self):
        """Test comparison between incompatible units."""
        a = Second(1.0)
        b = Byte(1.0)

        # These should return NotImplemented internally,
        # which Python will convert to False for equality
        assert not (a == b)
        assert a != b

        # For ordering comparisons, NotImplemented will raise TypeError
        with pytest.raises(TypeError):
            a < b
        with pytest.raises(TypeError):
            a > b


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_values(self):
        """Test with zero values."""
        zero_sec = Second(0.0)
        assert zero_sec.base_value == 0.0

        # Adding to zero
        result = zero_sec + Second(5.0)
        assert result.value == 5.0

        # Multiplying zero
        result = zero_sec * 10
        assert result.value == 0.0

    def test_negative_values(self):
        """Test with negative values (not typically valid for units, but should work mathematically)."""
        neg_flops = Flops(-1.0)
        pos_flops = Flops(3.0)

        result = pos_flops + neg_flops
        assert result.value == 2.0

        result = neg_flops * 3
        assert result.value == -3.0
