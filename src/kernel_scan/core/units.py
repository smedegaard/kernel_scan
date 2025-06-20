#!/usr/bin/env python3
"""
Units module for GPU performance analysis.

This module provides a set of units and unit conversions for GPU performance
analysis, including FLOPS, bytes, and seconds. It supports SI prefixes
and automatic unit conversion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Type, Union


@dataclass
class Dimension:
    """Represents the dimension of a physical quantity."""

    # The name of the dimension (e.g., "time", "memory", "compute")
    name: str

    def __str__(self) -> str:
        return self.name


class Prefix(Enum):
    """Metric prefixes for units."""

    PICO = 1e-12
    NANO = 1e-9
    MICRO = 1e-6
    MILLI = 1e-3
    NONE = 1.0
    KILO = 1e3
    MEGA = 1e6
    GIGA = 1e9
    TERA = 1e12
    PETA = 1e15
    EXA = 1e18

    def __init__(self, factor: float):
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
        """Get the value in the base unit (without prefix)."""
        return self.value * self.prefix.factor

    @property
    def name(self) -> str:
        """Get the full name of the unit including prefix."""
        if self.prefix == Prefix.NONE:
            return self.base_name
        return f"{self.prefix.name.lower()}{self.base_name}"

    @property
    def symbol(self) -> str:
        """Get the symbol for this unit, including prefix."""
        return f"{self.base_symbol}"

    def to(
        self, target_unit_class: Type["Unit"], prefix: Prefix = Prefix.NONE
    ) -> "Unit":
        """
        Convert to another unit with a specific prefix.

        Args:
            target_unit_class: The target unit class
            prefix: The target prefix (default: no prefix)

        Returns:
            The value in the target unit with the specified prefix
        """
        if not self._is_compatible_with(target_unit_class):
            raise ValueError(
                f"Cannot convert {self.__class__.__name__} to {target_unit_class.__name__}"
            )

        # Convert to the base value of the target unit
        base_val = self._convert_to_base_value(target_unit_class)

        # If target has specialized prefix subclasses, try to use them
        if target_unit_class == Flops:
            if prefix == Prefix.MEGA:
                return MegaFlops(base_val / Prefix.MEGA.factor)
            elif prefix == Prefix.GIGA:
                return GigaFlops(base_val / Prefix.GIGA.factor)
            elif prefix == Prefix.TERA:
                return TeraFlops(base_val / Prefix.TERA.factor)
            elif prefix == Prefix.PETA:
                return PetaFlops(base_val / Prefix.PETA.factor)
        elif target_unit_class == Byte:
            if prefix == Prefix.KILO:
                return KiloByte(base_val / Prefix.KILO.factor)
            elif prefix == Prefix.MEGA:
                return MegaByte(base_val / Prefix.MEGA.factor)
            elif prefix == Prefix.GIGA:
                return GigaByte(base_val / Prefix.GIGA.factor)
            elif prefix == Prefix.TERA:
                return TeraByte(base_val / Prefix.TERA.factor)
        elif target_unit_class == BytesPerSecond:
            if prefix == Prefix.GIGA:
                return GigaBytesPerSecond(base_val / Prefix.GIGA.factor)
            elif prefix == Prefix.TERA:
                return TeraBytePerSecond(base_val / Prefix.TERA.factor)

        # Default case: create a new instance with the given prefix
        new_val = base_val / prefix.factor
        return target_unit_class(new_val, prefix)

    def _is_compatible_with(self, target_unit_class: Type["Unit"]) -> bool:
        """Check if this unit can be converted to the target unit."""
        return self.dimension == target_unit_class.dimension

    def __add__(self, other: Union["Unit", float]) -> "Unit":
        """Add another unit or a scalar value."""
        if isinstance(other, float) or isinstance(other, int):
            # Check if this is a prefix-specific subclass
            if self.__class__.__init__ != Unit.__init__:
                # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
                return self.__class__(self.value + other)
            else:
                return self.__class__(self.value + other, self.prefix)

        if not isinstance(other, Unit):
            return NotImplemented

        if not self._is_compatible_with(other.__class__):
            raise ValueError(
                f"Cannot add {other.__class__.__name__} to {self.__class__.__name__}"
            )

        if not isinstance(other, self.__class__) or self.prefix != other.prefix:
            other = other.to(self.__class__, self.prefix)

        # Check if this is a prefix-specific subclass
        if self.__class__.__init__ != Unit.__init__:
            # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
            return self.__class__(self.value + other.value)
        else:
            return self.__class__(self.value + other.value, self.prefix)

    def __sub__(self, other: Union["Unit", float]) -> "Unit":
        """Subtract another unit or a scalar value."""
        if isinstance(other, float) or isinstance(other, int):
            # Check if this is a prefix-specific subclass
            if self.__class__.__init__ != Unit.__init__:
                # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
                return self.__class__(self.value - other)
            else:
                return self.__class__(self.value - other, self.prefix)

        if not isinstance(other, Unit):
            return NotImplemented

        if not self._is_compatible_with(other.__class__):
            raise ValueError(
                f"Cannot subtract {other.__class__.__name__} from {self.__class__.__name__}"
            )

        if not isinstance(other, self.__class__) or self.prefix != other.prefix:
            other = other.to(self.__class__, self.prefix)

        # Check if this is a prefix-specific subclass
        if self.__class__.__init__ != Unit.__init__:
            # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
            return self.__class__(self.value - other.value)
        else:
            return self.__class__(self.value - other.value, self.prefix)

    def __mul__(self, scalar: Union[float, int]) -> "Unit":
        """Multiply by a scalar value."""
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        # Check if this is a prefix-specific subclass
        if self.__class__.__init__ != Unit.__init__:
            # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
            return self.__class__(self.value * scalar)
        else:
            return self.__class__(self.value * scalar, self.prefix)

    def __rmul__(self, scalar: Union[float, int]) -> "Unit":
        """Multiply by a scalar value (from the right)."""
        return self.__mul__(scalar)

    def __truediv__(self, other: Union["Unit", float, int]) -> "Unit":
        """Divide by another unit or a scalar value."""
        if isinstance(other, float) or isinstance(other, int):
            # Check if this is a prefix-specific subclass
            if self.__class__.__init__ != Unit.__init__:
                # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
                return self.__class__(self.value / other)
            else:
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
            # Check if this is a prefix-specific subclass
            if self.__class__.__init__ != Unit.__init__:
                # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
                return self.__class__(
                    self.value / other.to(self.__class__, self.prefix).value
                )
            else:
                return self.__class__(
                    self.value / other.to(self.__class__, self.prefix).value,
                    self.prefix,
                )

        else:
            # For other cases, just return the value ratio
            # This is technically incorrect for incompatible dimensions, but allows flexibility
            # Check if this is a prefix-specific subclass
            if self.__class__.__init__ != Unit.__init__:
                # For prefix-specific classes like TeraFlops that don't accept a prefix parameter
                return self.__class__(self.value / other.value)
            else:
                return self.__class__(self.value / other.value, self.prefix)

    @abstractmethod
    def _convert_to_base_value(self, target_unit_class: Type["Unit"]) -> float:
        """Convert the base value to the target unit's base value."""
        pass

    def with_prefix(self, prefix: Prefix) -> "Unit":
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

    def to_pico(self) -> "Unit":
        """Convert to pico prefix."""
        return self.with_prefix(Prefix.PICO)

    def to_nano(self) -> "Unit":
        """Convert to nano prefix."""
        return self.with_prefix(Prefix.NANO)

    def to_micro(self) -> "Unit":
        """Convert to micro prefix."""
        return self.with_prefix(Prefix.MICRO)

    def to_milli(self) -> "Unit":
        """Convert to milli prefix."""
        return self.with_prefix(Prefix.MILLI)

    def to_base(self) -> "Unit":
        """Convert to base unit (no prefix)."""
        return self.with_prefix(Prefix.NONE)

    def to_kilo(self) -> "Unit":
        """Convert to kilo prefix."""
        return self.with_prefix(Prefix.KILO)

    def to_mega(self) -> "Unit":
        """Convert to mega prefix."""
        return self.with_prefix(Prefix.MEGA)

    def to_giga(self) -> "Unit":
        """Convert to giga prefix."""
        return self.with_prefix(Prefix.GIGA)

    def to_tera(self) -> "Unit":
        """Convert to tera prefix."""
        return self.with_prefix(Prefix.TERA)

    def to_peta(self) -> "Unit":
        """Convert to peta prefix."""
        return self.with_prefix(Prefix.PETA)

    def to_exa(self) -> "Unit":
        """Convert to exa prefix."""
        return self.with_prefix(Prefix.EXA)

    def __str__(self) -> str:
        return f"{self.value} {self.symbol}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value}, Prefix.{self.prefix.name})"

    def __eq__(self, other: Any) -> bool:
        """Check if two units are equal (have the same base value)."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return False
        return abs(self.base_value - other.base_value) < 1e-10

    def __lt__(self, other: "Unit") -> bool:
        """Check if this unit is less than another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value < other.base_value

    def __gt__(self, other: "Unit") -> bool:
        """Check if this unit is greater than another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value > other.base_value

    def __le__(self, other: "Unit") -> bool:
        """Check if this unit is less than or equal to another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value <= other.base_value

    def __ge__(self, other: "Unit") -> bool:
        """Check if this unit is greater than or equal to another."""
        if not isinstance(other, Unit) or not self._is_compatible_with(other.__class__):
            return NotImplemented
        return self.base_value >= other.base_value

    def format(self, precision: int = 2) -> str:
        """Format the unit value with specified precision."""
        return f"{self.value:.{precision}f} {self.symbol}"


class Second(Unit):
    """Represents seconds."""

    dimension = Dimension("time")
    base_name = "second"
    base_symbol = "s"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert seconds to the target unit's base value."""
        return self.base_value


class Millisecond(Second):
    """Represents milliseconds (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MILLI)


class Microsecond(Second):
    """Represents microseconds (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MICRO)


class Bit(Unit):
    """Represents bits."""

    dimension = Dimension("memory")
    base_name = "bit"
    base_symbol = "bit"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert bits to the target unit's base value."""
        if target_unit_class == Byte:
            return self.base_value / 8
        return self.base_value


class Byte(Unit):
    """Represents bytes."""

    dimension = Dimension("memory")
    base_name = "byte"
    base_symbol = "B"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert bytes to the target unit's base value."""
        if target_unit_class == Bit:
            return self.base_value * 8
        return self.base_value


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


class Flop(Unit):
    """Represents floating-point operations."""

    dimension = Dimension("compute")
    base_name = "flop"
    base_symbol = "FLOP"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert FLOPs to the target unit's base value."""
        return self.base_value


class Flops(Unit):
    """Represents floating-point operations per second."""

    dimension = Dimension("compute_rate")
    base_name = "flops"
    base_symbol = "FLOPS"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert FLOPS to the target unit's base value."""
        return self.base_value


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


class BytesPerSecond(Unit):
    """Represents bytes per second (bandwidth)."""

    dimension = Dimension("bandwidth")
    base_name = "bytes per second"
    base_symbol = "B/s"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert bytes per second to the target unit's base value."""
        if target_unit_class == BitsPerSecond:
            return self.base_value * 8
        return self.base_value


class BitsPerSecond(Unit):
    """Represents bits per second (bandwidth)."""

    dimension = Dimension("bandwidth")
    base_name = "bits per second"
    base_symbol = "bit/s"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert bits per second to the target unit's base value."""
        if target_unit_class == BytesPerSecond:
            return self.base_value / 8
        return self.base_value


class GigaBytesPerSecond(BytesPerSecond):
    """Represents gigabytes per second (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.GIGA)


class TeraBytePerSecond(BytesPerSecond):
    """Represents terabytes per second (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.TERA)


class FlopsPerByte(Unit):
    """Represents FLOPS per byte (arithmetic intensity)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "flops per byte"
    base_symbol = "FLOPS/B"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert FLOPS per byte to the target unit's base value."""
        if target_unit_class == FlopsPerBit:
            return self.base_value / 8
        return self.base_value


class FlopsPerBit(Unit):
    """Represents FLOPS per bit (arithmetic intensity)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "flops per bit"
    base_symbol = "FLOPS/bit"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert FLOPS per bit to the target unit's base value."""
        if target_unit_class == FlopsPerByte:
            return self.base_value * 8
        return self.base_value


# INTEGER OPERATIONS CLASSES (IOPS)


class Iop(Unit):
    """Represents integer operations."""

    dimension = Dimension("compute")
    base_name = "iop"
    base_symbol = "IOP"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert IOPs to the target unit's base value."""
        return self.base_value


class Iops(Unit):
    """Represents integer operations per second."""

    dimension = Dimension("compute_rate")
    base_name = "iops"
    base_symbol = "IOPS"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert IOPS to the target unit's base value."""
        return self.base_value


class MegaIops(Iops):
    """Represents megaIOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MEGA)


class GigaIops(Iops):
    """Represents gigaIOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.GIGA)


class TeraIops(Iops):
    """Represents teraIOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.TERA)


class PetaIops(Iops):
    """Represents petaIOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.PETA)


# ARITHMETIC INTENSITY CLASSES FOR INTEGER OPERATIONS


class IopsPerByte(Unit):
    """Represents IOPS per byte (arithmetic intensity for integer operations)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "iops per byte"
    base_symbol = "IOPS/B"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert IOPS per byte to the target unit's base value."""
        if target_unit_class == IopsPerBit:
            return self.base_value / 8
        return self.base_value


class IopsPerBit(Unit):
    """Represents IOPS per bit (arithmetic intensity for integer operations)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "iops per bit"
    base_symbol = "IOPS/bit"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert IOPS per bit to the target unit's base value."""
        if target_unit_class == IopsPerByte:
            return self.base_value * 8
        return self.base_value


# GENERIC OPERATIONS CLASSES (for mixed workloads)


class Op(Unit):
    """Represents generic operations (floating-point, integer, or mixed)."""

    dimension = Dimension("compute")
    base_name = "op"
    base_symbol = "OP"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert operations to the target unit's base value."""
        return self.base_value


class Ops(Unit):
    """Represents generic operations per second."""

    dimension = Dimension("compute_rate")
    base_name = "ops"
    base_symbol = "OPS"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert OPS to the target unit's base value."""
        return self.base_value


class MegaOps(Ops):
    """Represents megaOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.MEGA)


class GigaOps(Ops):
    """Represents gigaOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.GIGA)


class TeraOps(Ops):
    """Represents teraOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.TERA)


class PetaOps(Ops):
    """Represents petaOPS (convenience class)."""

    def __init__(self, value: float):
        super().__init__(value, Prefix.PETA)


# ARITHMETIC INTENSITY CLASSES FOR GENERIC OPERATIONS


class OpsPerByte(Unit):
    """Represents operations per byte (generic arithmetic intensity)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "operations per byte"
    base_symbol = "OPS/B"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert operations per byte to the target unit's base value."""
        if target_unit_class == OpsPerBit:
            return self.base_value / 8
        elif target_unit_class == FlopsPerByte:
            return self.base_value  # Assuming 1:1 conversion for compatibility
        elif target_unit_class == IopsPerByte:
            return self.base_value  # Assuming 1:1 conversion
        return self.base_value


class OpsPerBit(Unit):
    """Represents operations per bit (generic arithmetic intensity)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "operations per bit"
    base_symbol = "OPS/bit"

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert operations per bit to the target unit's base value."""
        if target_unit_class == OpsPerByte:
            return self.base_value * 8
        elif target_unit_class == FlopsPerBit:
            return self.base_value  # Assuming 1:1 conversion for compatibility
        elif target_unit_class == IopsPerBit:
            return self.base_value  # Assuming 1:1 conversion
        return self.base_value
