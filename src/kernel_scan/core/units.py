#!/usr/bin/env python3
"""
Units module for GPU performance analysis.

This module provides a set of units and unit conversions for GPU performance
analysis, including FLOPS, bytes, and seconds. It supports SI prefixes
and automatic unit conversion.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Optional, Type, Union


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
        prefix_symbols = {
            Prefix.PICO: "p",
            Prefix.NANO: "n",
            Prefix.MICRO: "μ",
            Prefix.MILLI: "m",
            Prefix.NONE: "",
            Prefix.KILO: "k",
            Prefix.MEGA: "M",
            Prefix.GIGA: "G",
            Prefix.TERA: "T",
            Prefix.PETA: "P",
            Prefix.EXA: "E",
        }
        prefix_symbol = prefix_symbols.get(self.prefix, "")
        return f"{prefix_symbol}{self.base_symbol}"

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
        scaled_val = base_val / target_unit_class._prefix.value
        target = target_unit_class(scaled_val)
        return target

    def with_prefix(self, prefix: Prefix) -> "Unit":
        """
        Create a new instance with the same value but a different prefix.

        Args:
            prefix: The new prefix to use

        Returns:
            A new unit instance with the adjusted value
        """
        base_val = self.base_value
        scaled_val = base_val / prefix.factor
        base_class = self._get_base_class()

        # Use convenience class if available
        return self._create_convenience_instance(
            base_class, prefix, scaled_val
        ) or self.__class__(scaled_val, prefix)

    def _create_convenience_instance(
        self, unit_class: Type["Unit"], prefix: Prefix, value: float
    ) -> Optional["Unit"]:
        """Create a convenience class instance if one exists for the given unit class and prefix."""
        # Find all Unit subclasses that have the _prefix attribute
        convenience_classes = [
            cls
            for cls in globals().values()
            if isinstance(cls, type)
            and issubclass(cls, Unit)
            and cls != Unit
            and hasattr(cls, "_prefix")
            and cls._prefix == prefix
        ]

        # Check which ones have the specified base class
        for cls in convenience_classes:
            if unit_class in cls.__mro__:
                return cls(value)

        return None

    def _get_base_class(self) -> Type["Unit"]:
        """Get the base class for this unit."""
        # For convenience classes, we want the first non-convenience base class
        for base in self.__class__.__mro__:
            if base != self.__class__ and base != Unit and issubclass(base, Unit):
                # Found a base class that is a Unit but not the convenience class itself
                if base.__init__ == Unit.__init__:
                    # This is a base class with the standard Unit.__init__
                    return base

        # If no suitable base class is found, return the class itself
        return self.__class__

    # def _get_base_class(self) -> Type["Unit"]:
    #     """Get the base class for this unit."""
    #     base_class_map = {
    #         MegaFlops: Flops,
    #         GigaFlops: Flops,
    #         TeraFlops: Flops,
    #         PetaFlops: Flops,
    #         KiloByte: Byte,
    #         MegaByte: Byte,
    #         GigaByte: Byte,
    #         TeraByte: Byte,
    #         GigaBytesPerSecond: BytesPerSecond,
    #         TeraBytePerSecond: BytesPerSecond,
    #         MilliSecond: Second,
    #         MicroSecond: Second,
    #     }
    #     return base_class_map.get(self.__class__, self.__class__)

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

    def _convert_to_base_value(self, target_unit_class: Type["Unit"]) -> float:
        """Convert the base value to the target unit's base value."""
        # By default, if dimensions match, the base value stays the same
        if self.dimension == target_unit_class.dimension:
            return self.base_value

        # For cross-dimension conversions, subclasses should override this method
        # to provide specific conversion logic
        raise ValueError(
            f"Cannot convert from {self.__class__.__name__} ({self.dimension.name}) "
            f"to {target_unit_class.__name__} ({target_unit_class.dimension.name})"
        )

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
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert seconds to the target unit's base value."""
        return self.base_value


class MilliSecond(Second):
    """Represents milliseconds (convenience class)."""

    _prefix = Prefix.MILLI

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class MicroSecond(Second):
    """Represents microseconds (convenience class)."""

    _prefix = Prefix.MICRO

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class Bit(Unit):
    """Represents bits."""

    dimension = Dimension("memory")
    base_name = "bit"
    base_symbol = "bit"
    _prefix = Prefix.NONE

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
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert bytes to the target unit's base value."""
        if target_unit_class == Bit:
            return self.base_value * 8
        return self.base_value


class KiloByte(Byte):
    """Represents kilobytes (convenience class)."""

    _prefix = Prefix.KILO

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class MegaByte(Byte):
    """Represents megabytes (convenience class)."""

    _prefix = Prefix.MEGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class GigaByte(Byte):
    """Represents gigabytes (convenience class)."""

    _prefix = Prefix.GIGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class TeraByte(Byte):
    """Represents terabytes (convenience class)."""

    _prefix = Prefix.TERA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class Flop(Unit):
    """Represents floating-point operations."""

    dimension = Dimension("compute")
    base_name = "flop"
    base_symbol = "FLOP"
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert FLOPs to the target unit's base value."""
        return self.base_value


class Flops(Unit):
    """Represents floating-point operations per second."""

    dimension = Dimension("compute_rate")
    base_name = "flops"
    base_symbol = "FLOPS"
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert FLOPS to the target unit's base value."""
        return self.base_value


class MegaFlops(Flops):
    """Represents megaFLOPS (convenience class)."""

    _prefix = Prefix.MEGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class GigaFlops(Flops):
    """Represents gigaFLOPS (convenience class)."""

    _prefix = Prefix.GIGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class TeraFlops(Flops):
    """Represents teraFLOPS (convenience class)."""

    _prefix = Prefix.TERA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class PetaFlops(Flops):
    """Represents petaFLOPS (convenience class)."""

    _prefix = Prefix.PETA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class BytesPerSecond(Unit):
    """Represents bytes per second (bandwidth)."""

    dimension = Dimension("bandwidth")
    base_name = "bytes per second"
    base_symbol = "B/s"
    _prefix = Prefix.NONE

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
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert bits per second to the target unit's base value."""
        if target_unit_class == BytesPerSecond:
            return self.base_value / 8
        return self.base_value


class GigaBytesPerSecond(BytesPerSecond):
    """Represents gigabytes per second (convenience class)."""

    _prefix = Prefix.GIGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class TeraBytePerSecond(BytesPerSecond):
    """Represents terabytes per second (convenience class)."""

    _prefix = Prefix.TERA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class FlopsPerByte(Unit):
    """Represents FLOPS per byte (arithmetic intensity)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "flops per byte"
    base_symbol = "FLOPS/B"
    _prefix = Prefix.NONE

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
    _prefix = Prefix.NONE

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
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert IOPs to the target unit's base value."""
        return self.base_value


class Iops(Unit):
    """Represents integer operations per second."""

    dimension = Dimension("compute_rate")
    base_name = "iops"
    base_symbol = "IOPS"
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert IOPS to the target unit's base value."""
        return self.base_value


class MegaIops(Iops):
    """Represents megaIOPS (convenience class)."""

    _prefix = Prefix.MEGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class GigaIops(Iops):
    """Represents gigaIOPS (convenience class)."""

    _prefix = Prefix.GIGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class TeraIops(Iops):
    """Represents teraIOPS (convenience class)."""

    _prefix = Prefix.TERA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class PetaIops(Iops):
    """Represents petaIOPS (convenience class)."""

    _prefix = Prefix.PETA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


# ARITHMETIC INTENSITY CLASSES FOR INTEGER OPERATIONS


class IopsPerByte(Unit):
    """Represents IOPS per byte (arithmetic intensity for integer operations)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "iops per byte"
    base_symbol = "IOPS/B"
    _prefix = Prefix.NONE

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
    _prefix = Prefix.NONE

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
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert operations to the target unit's base value."""
        return self.base_value


class Ops(Unit):
    """Represents generic operations per second."""

    dimension = Dimension("compute_rate")
    base_name = "ops"
    base_symbol = "OPS"
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert OPS to the target unit's base value."""
        return self.base_value


class MegaOps(Ops):
    """Represents megaOPS (convenience class)."""

    _prefix = Prefix.MEGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class GigaOps(Ops):
    """Represents gigaOPS (convenience class)."""

    _prefix = Prefix.GIGA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class TeraOps(Ops):
    """Represents teraOPS (convenience class)."""

    _prefix = Prefix.TERA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


class PetaOps(Ops):
    """Represents petaOPS (convenience class)."""

    _prefix = Prefix.PETA

    def __init__(self, value: float):
        super().__init__(value, self._prefix)


# ARITHMETIC INTENSITY CLASSES FOR GENERIC OPERATIONS


class OpsPerByte(Unit):
    """Represents operations per byte (generic arithmetic intensity)."""

    dimension = Dimension("arithmetic_intensity")
    base_name = "operations per byte"
    base_symbol = "OPS/B"
    _prefix = Prefix.NONE

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
    _prefix = Prefix.NONE

    def _convert_to_base_value(self, target_unit_class: Type[Unit]) -> float:
        """Convert operations per bit to the target unit's base value."""
        if target_unit_class == OpsPerByte:
            return self.base_value * 8
        elif target_unit_class == FlopsPerBit:
            return self.base_value  # Assuming 1:1 conversion for compatibility
        elif target_unit_class == IopsPerBit:
            return self.base_value  # Assuming 1:1 conversion
        return self.base_value


CONVENIENCE_CLASSES = {
    # Time units
    MicroSecond: Second,
    MilliSecond: Second,
    # Storage units
    KiloByte: Byte,
    MegaByte: Byte,
    GigaByte: Byte,
    TeraByte: Byte,
    # Computing performance units
    MegaFlops: Flops,
    GigaFlops: Flops,
    TeraFlops: Flops,
    PetaFlops: Flops,
    # Bandwidth units
    GigaBytesPerSecond: BytesPerSecond,
    TeraBytePerSecond: BytesPerSecond,
    # I/O operations units
    MegaIops: Iops,
    GigaIops: Iops,
    TeraIops: Iops,
    PetaIops: Iops,
    # General operations units
    MegaOps: Ops,
    GigaOps: Ops,
    TeraOps: Ops,
    PetaOps: Ops,
}
