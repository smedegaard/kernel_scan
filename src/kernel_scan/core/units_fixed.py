"""
Measurement units for GPU performance profiling.

This module provides a clean, ergonomic API for working with units related to
GPU performance profiling, such as FLOPS, bytes, and time measurements. It handles
unit prefixes, conversions, and tracking of unit dimensions.

Example usage:
    # Create units with specific prefixes
    compute = TeraFlops(1.2)
    bandwidth = GigaBytesPerSecond(900)

    # Convert between units
    gflops = compute.to_giga()  # 1200 GFLOPs

    # Get arithmetic intensity
    intensity = compute / bandwidth  # FLOPS/B
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, ClassVar, Type, Union


class Dimension(Enum):
    """Defines the physical dimension of a unit."""

    TIME = auto()
    COMPUTATION = auto()
    DATA = auto()
    COMPUTE_PERFORMANCE = auto()
    DATA_BANDWIDTH = auto()
    ARITHMETIC_INTENSITY = auto()

    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")


class Prefix(Enum):
    """Metric prefixes for units."""

    PICO = ("p", 1e-12)
    NANO = ("n", 1e-9)
    MICRO = ("μ", 1e-6)
    MILLI = ("m", 1e-3)
    NONE = ("", 1.0)
    KILO = ("K", 1e3)
    MEGA = ("M", 1e6)
    GIGA = ("G", 1e9)
    TERA = ("T", 1e12)
    PETA = ("P", 1e15)
    EXA = ("E", 1e18)

    def __init__(self, symbol: str, factor: float):
        self.symbol = symbol
        self.factor = factor


class Unit(ABC):
    """Base class for all measurement units."""

    dimension: ClassVar[Dimension]
    base_name: ClassVar[str]
    base_symbol: ClassVar[str]

    def __init__(self, value: float, prefix: Prefix = Prefix.NONE):
        """
        Create a new unit instance.

        Args:
            value: The numeric value in the given prefix scale
            prefix: The metric prefix to use (default: no prefix)
        """
        self.value = value
        self.prefix = prefix

    @property
    def base_value(self) -> float:
        """Value converted to the base unit without prefix."""
        return self.value * self.prefix.factor

    @property
    def name(self) -> str:
        """Full name of the unit with prefix."""
        if self.prefix == Prefix.NONE:
            return self.base_name
        prefix_name = self.prefix.name.lower().capitalize()
        return f"{prefix_name}{self.base_name}"

    @property
    def symbol(self) -> str:
        """Symbol for the unit with prefix."""
        return f"{self.prefix.symbol}{self.base_symbol}"

    def to(self, target_unit_class: Type[Unit], prefix: Prefix = Prefix.NONE) -> Unit:
        """
        Convert to another unit with optional prefix.

        Args:
            target_unit_class: The target unit class
            prefix: The prefix to use for the target unit

        Returns:
            A new instance of the target unit class

        Raises:
            ValueError: If the units have incompatible dimensions
        """
        if not self._is_compatible_with(target_unit_class):
            raise ValueError(
                f"Cannot convert {self.__class__.__name__} to {target_unit_class.__name__}: "
                f"incompatible dimensions ({self.dimension} vs {target_unit_class.dimension})"
            )

        target_value_base = self._convert_to_base_value(target_unit_class)
        target_value = target_value_base / prefix.factor

        return target_unit_class(target_value, prefix)

    def _is_compatible_with(self, target_unit_class: Type[Unit]) -> bool:
        """Check if this unit can be converted to the target unit."""
        return self.dimension == target_unit_class.dimension

    @abstractmethod
    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert the base value to the target unit's base value."""
        pass

    def with_prefix(self, prefix: Prefix) -> Unit:
        """
        Create a new instance with the same value but a different prefix.

        Args:
            prefix: The new prefix to use

        Returns:
            A new unit instance with the adjusted value
        """
        base_val = self.base_value

        # Special handling for unit classes with prefix-specific subclasses
        if isinstance(self, Flops):
            if prefix == Prefix.MEGA:
                return MegaFlops(base_val / Prefix.MEGA.factor)
            elif prefix == Prefix.GIGA:
                return GigaFlops(base_val / Prefix.GIGA.factor)
            elif prefix == Prefix.TERA:
                return TeraFlops(base_val / Prefix.TERA.factor)
            elif prefix == Prefix.PETA:
                return PetaFlops(base_val / Prefix.PETA.factor)
        elif isinstance(self, Byte):
            if prefix == Prefix.KILO:
                return KiloByte(base_val / Prefix.KILO.factor)
            elif prefix == Prefix.MEGA:
                return MegaByte(base_val / Prefix.MEGA.factor)
            elif prefix == Prefix.GIGA:
                return GigaByte(base_val / Prefix.GIGA.factor)
            elif prefix == Prefix.TERA:
                return TeraByte(base_val / Prefix.TERA.factor)
        elif isinstance(self, BytesPerSecond):
            if prefix == Prefix.GIGA:
                return GigaBytesPerSecond(base_val / Prefix.GIGA.factor)
            elif prefix == Prefix.TERA:
                return TeraBytePerSecond(base_val / Prefix.TERA.factor)

        # For other cases, use the standard approach
        new_val = base_val / prefix.factor
        return self.__class__(new_val, prefix)

    # Convenience methods for common prefix conversions
    def to_pico(self) -> Unit:
        """Convert to pico prefix."""
        return self.with_prefix(Prefix.PICO)

    def to_nano(self) -> Unit:
        """Convert to nano prefix."""
        return self.with_prefix(Prefix.NANO)

    def to_micro(self) -> Unit:
        """Convert to micro prefix."""
        return self.with_prefix(Prefix.MICRO)

    def to_milli(self) -> Unit:
        """Convert to milli prefix."""
        return self.with_prefix(Prefix.MILLI)

    def to_base(self) -> Unit:
        """Convert to base unit (no prefix)."""
        return self.with_prefix(Prefix.NONE)

    def to_kilo(self) -> Unit:
        """Convert to kilo prefix."""
        return self.with_prefix(Prefix.KILO)

    def to_mega(self) -> Unit:
        """Convert to mega prefix."""
        return self.with_prefix(Prefix.MEGA)

    def to_giga(self) -> Unit:
        """Convert to giga prefix."""
        return self.with_prefix(Prefix.GIGA)

    def to_tera(self) -> Unit:
        """Convert to tera prefix."""
        return self.with_prefix(Prefix.TERA)

    def to_peta(self) -> Unit:
        """Convert to peta prefix."""
        return self.with_prefix(Prefix.PETA)

    def to_exa(self) -> Unit:
        """Convert to exa prefix."""
        return self.with_prefix(Prefix.EXA)

    def __str__(self) -> str:
        return f"{self.value} {self.symbol}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value}, Prefix.{self.prefix.name})"

    def __add__(self, other: Union[Unit, float]) -> Unit:
        """Add another unit or a scalar value."""
        if isinstance(other, float) or isinstance(other, int):
            return self.__class__(self.value + other, self.prefix)

        if not isinstance(other, Unit):
            return NotImplemented

        if not self._is_compatible_with(other.__class__):
            raise ValueError(
                f"Cannot add {other.__class__.__name__} to {self.__class__.__name__}"
            )

        if not isinstance(other, self.__class__) or self.prefix != other.prefix:
            other = other.to(self.__class__, self.prefix)

        return self.__class__(self.value + other.value, self.prefix)

    def __sub__(self, other: Union[Unit, float]) -> Unit:
        """Subtract another unit or a scalar value."""
        if isinstance(other, float) or isinstance(other, int):
            return self.__class__(self.value - other, self.prefix)

        if not isinstance(other, Unit):
            return NotImplemented

        if not self._is_compatible_with(other.__class__):
            raise ValueError(
                f"Cannot subtract {other.__class__.__name__} from {self.__class__.__name__}"
            )

        if not isinstance(other, self.__class__) or self.prefix != other.prefix:
            other = other.to(self.__class__, self.prefix)

        return self.__class__(self.value - other.value, self.prefix)

    def __mul__(self, scalar: Union[float, int]) -> Unit:
        """Multiply by a scalar value."""
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented
        return self.__class__(self.value * scalar, self.prefix)

    def __rmul__(self, scalar: Union[float, int]) -> Unit:
        """Multiply by a scalar value (from the right)."""
        return self.__mul__(scalar)

    def __truediv__(self, other: Union[Unit, float, int]) -> Unit:
        """Divide by another unit or a scalar value."""
        if isinstance(other, float) or isinstance(other, int):
            return self.__class__(self.value / other, self.prefix)

        if not isinstance(other, Unit):
            return NotImplemented

        # Handle specific unit division cases
        if isinstance(self, Flops) and isinstance(other, BytesPerSecond):
            # FLOPS / BytesPerSecond = FLOPS/Byte (arithmetic intensity)
            return FlopsPerByte(self.base_value / other.base_value)

        elif isinstance(self, Flop) and isinstance(other, Second):
            # FLOP / Second = FLOPS (compute performance)
            return Flops(self.base_value / other.base_value)

        elif isinstance(self, Byte) and isinstance(other, Second):
            # Byte / Second = BytesPerSecond (bandwidth)
            return BytesPerSecond(self.base_value / other.base_value)

        elif isinstance(self, Bit) and isinstance(other, Second):
            # Bit / Second = BitsPerSecond (bandwidth)
            return BitsPerSecond(self.base_value / other.base_value)

        elif self.dimension == other.dimension:
            # Same dimension means dimensionless result
            return self.__class__(
                self.value / other.to(self.__class__, self.prefix).value, self.prefix
            )

        else:
            # For other cases, just return the value ratio
            # This is technically incorrect for incompatible dimensions, but allows flexibility
            return self.__class__(self.value / other.value, self.prefix)

    def __eq__(self, other: Any) -> bool:
        """Check if two units are equal (have the same base value)."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return False
        return abs(self.base_value - other.base_value) < 1e-10

    def __lt__(self, other: Unit) -> bool:
        """Check if this unit is less than another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value < other.base_value

    def __gt__(self, other: Unit) -> bool:
        """Check if this unit is greater than another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value > other.base_value

    def __le__(self, other: Unit) -> bool:
        """Check if this unit is less than or equal to another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value <= other.base_value

    def __ge__(self, other: Unit) -> bool:
        """Check if this unit is greater than or equal to another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value >= other.base_value

    def format(self, precision: int = 2) -> str:
        """Format the unit with specified precision."""
        return f"{self.value:.{precision}f} {self.symbol}"


# Time units
class Second(Unit):
    """Represents a time unit in seconds."""

    dimension = Dimension.TIME
    base_name = "second"
    base_symbol = "s"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        # All time units use second as the base
        return self.base_value


class Millisecond(Second):
    """Represents milliseconds (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MILLI)


class Microsecond(Second):
    """Represents microseconds (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MICRO)


# Data units
class Bit(Unit):
    """Represents a data unit in bits."""

    dimension = Dimension.DATA
    base_name = "bit"
    base_symbol = "bit"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        if target_unit_class == Byte:
            return self.base_value / 8
        return self.base_value


class Byte(Unit):
    """Represents a data unit in bytes."""

    dimension = Dimension.DATA
    base_name = "byte"
    base_symbol = "B"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        if target_unit_class == Bit:
            return self.base_value * 8
        return self.base_value


# Convenience classes for common data units
class KiloByte(Byte):
    """Represents kilobytes (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.KILO)


class MegaByte(Byte):
    """Represents megabytes (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MEGA)


class GigaByte(Byte):
    """Represents gigabytes (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.GIGA)


class TeraByte(Byte):
    """Represents terabytes (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.TERA)


# Computation units
class Flop(Unit):
    """Represents a floating-point operation."""

    dimension = Dimension.COMPUTATION
    base_name = "flop"
    base_symbol = "FLOP"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        return self.base_value


# Compute performance units
class Flops(Unit):
    """Represents floating-point operations per second."""

    dimension = Dimension.COMPUTE_PERFORMANCE
    base_name = "flops"
    base_symbol = "FLOPS"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        return self.base_value


# Convenience classes for common FLOPS units
class MegaFlops(Flops):
    """Represents megaFLOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MEGA)


class GigaFlops(Flops):
    """Represents gigaFLOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.GIGA)


class TeraFlops(Flops):
    """Represents teraFLOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.TERA)


class PetaFlops(Flops):
    """Represents petaFLOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.PETA)


# Data bandwidth units
class BytesPerSecond(Unit):
    """Represents data bandwidth in bytes per second."""

    dimension = Dimension.DATA_BANDWIDTH
    base_name = "bytes_per_second"
    base_symbol = "B/s"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        if target_unit_class == BitsPerSecond:
            return self.base_value * 8
        return self.base_value


class BitsPerSecond(Unit):
    """Represents data bandwidth in bits per second."""

    dimension = Dimension.DATA_BANDWIDTH
    base_name = "bits_per_second"
    base_symbol = "bit/s"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        if target_unit_class == BytesPerSecond:
            return self.base_value / 8
        return self.base_value


# Convenience classes for common bandwidth units
class GigaBytesPerSecond(BytesPerSecond):
    """Represents gigabytes per second (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.GIGA)


class TeraBytePerSecond(BytesPerSecond):
    """Represents terabytes per second (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.TERA)


# Arithmetic intensity units
class FlopsPerByte(Unit):
    """Represents arithmetic intensity in FLOPS per byte."""

    dimension = Dimension.ARITHMETIC_INTENSITY
    base_name = "flops_per_byte"
    base_symbol = "FLOPS/B"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        if target_unit_class == FlopsPerBit:
            return self.base_value / 8
        return self.base_value


class FlopsPerBit(Unit):
    """Represents arithmetic intensity in FLOPS per bit."""

    dimension = Dimension.ARITHMETIC_INTENSITY
    base_name = "flops_per_bit"
    base_symbol = "FLOPS/bit"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        if target_unit_class == FlopsPerByte:
            return self.base_value * 8
        return self.base_value


# Factory functions for common operations
def compute_arithmetic_intensity(
    compute: Flops, bandwidth: BytesPerSecond
) -> FlopsPerByte:
    """
    Calculate arithmetic intensity from compute performance and memory bandwidth.

    Args:
        compute: Compute performance in FLOPS
        bandwidth: Memory bandwidth in bytes per second

    Returns:
        Arithmetic intensity in FLOPS per byte
    """
    # Use the division operator directly, which now returns FlopsPerByte
    return compute / bandwidth


def peak_performance(
    compute: Flops, bandwidth: BytesPerSecond, arithmetic_intensity: FlopsPerByte
) -> Flops:
    """
    Calculate attainable peak performance based on roofline model.

    Args:
        compute: Peak compute performance
        bandwidth: Peak memory bandwidth
        arithmetic_intensity: Arithmetic intensity of the operation

    Returns:
        Attainable performance in FLOPS
    """
    memory_bound = bandwidth.base_value * arithmetic_intensity.base_value
    attainable = min(compute.base_value, memory_bound)

    # Return with the same prefix as compute
    return Flops(attainable / compute.prefix.factor, compute.prefix)
