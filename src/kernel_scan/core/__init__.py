"""
Core components for kernel profiling.

This package contains the core abstractions and data types used
throughout the kernel_scan library.
"""

# Import types module to make it accessible
# Import units module to make it accessible
from kernel_scan.core import types, units

# Import specs for convenience
from kernel_scan.core.specs import (
    AcceleratorSpec,
    KernelSpec,
    KernelSpecBuilder,
    TensorSpec,
)

# Import key unit classes for convenience
from kernel_scan.core.units import (
    # Data units
    Bit,
    BitsPerSecond,
    Byte,
    # Bandwidth units
    BytesPerSecond,
    # Base classes
    Dimension,
    # Computation units
    Flop,
    Flops,
    FlopsPerBit,
    # Arithmetic intensity units
    FlopsPerByte,
    GigaByte,
    GigaBytesPerSecond,
    GigaFlops,
    KiloByte,
    MegaByte,
    MegaFlops,
    Microsecond,
    Millisecond,
    PetaFlops,
    Prefix,
    # Time units
    Second,
    TeraByte,
    TeraBytePerSecond,
    TeraFlops,
    Unit,
)
