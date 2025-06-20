# Units Module for GPU Performance Profiling

A clean, ergonomic API for working with measurement units related to GPU performance profiling.

## Motivation

This module provides a more ergonomic alternative to libraries like `pint` for the specific domain of GPU performance profiling. Key design goals include:

- Simple, intuitive API without using multiplication operator `*` for unit specification
- Built-in support for common units in GPU profiling (FLOPS, bytes, seconds)
- First-class support for unit prefixes (kilo, mega, giga, tera, etc.)
- Dimension tracking to prevent invalid operations
- Easy conversion between units and prefixes

## Basic Usage

```python
from kernel_scan.core.units import TeraFlops, GigaBytesPerSecond, compute_arithmetic_intensity

# Create performance metrics
compute_perf = TeraFlops(1.2)  # 1.2 TFLOPS
bandwidth = GigaBytesPerSecond(900)  # 900 GB/s

# Convert between units with different prefixes
gflops = compute_perf.to_giga()  # 1200 GFLOPs
print(gflops)  # "1200.0 GFLOPS"

# Calculate arithmetic intensity
intensity = compute_arithmetic_intensity(compute_perf, bandwidth)
print(intensity)  # "0.00133 FLOPS/B"

# Apply the roofline model
from kernel_scan.core.units import peak_performance, FlopsPerByte

# Determine if an operation is compute-bound or memory-bound
ai = FlopsPerByte(0.5)  # Arithmetic intensity of 0.5 FLOPS/B
max_perf = peak_performance(compute_perf, bandwidth, ai)
print(max_perf)  # The attainable performance (minimum of compute peak and memory constraint)
```

## Available Units

### Time Units
- `Second` - Base unit for time
- `Millisecond` - Convenience class for milliseconds
- `Microsecond` - Convenience class for microseconds

### Data Units
- `Bit` - Base unit for data (bits)
- `Byte` - Base unit for data (bytes)
- `KiloByte`, `MegaByte`, `GigaByte`, `TeraByte` - Convenience classes

### Compute Units
- `Flop` - Floating-point operation
- `Flops` - Floating-point operations per second
- `MegaFlops`, `GigaFlops`, `TeraFlops`, `PetaFlops` - Convenience classes

### Bandwidth Units
- `BytesPerSecond` - Base unit for data bandwidth
- `BitsPerSecond` - Alternative base unit for data bandwidth
- `GigaBytesPerSecond`, `TeraBytePerSecond` - Convenience classes

### Arithmetic Intensity Units
- `FlopsPerByte` - Base unit for arithmetic intensity
- `FlopsPerBit` - Alternative unit for arithmetic intensity

## Features

### Unit Conversion

Convert between different prefixes of the same unit:

```python
from kernel_scan.core.units import TeraFlops, Prefix

# Create a TeraFlops object
tflops = TeraFlops(1.5)  # 1.5 TFLOPS

# Convert to other prefixes
gflops = tflops.to_giga()  # 1500 GFLOPS
mflops = tflops.to_mega()  # 1500000 MFLOPS

# Or use the with_prefix method
pflops = tflops.with_prefix(Prefix.PETA)  # 0.0015 PFLOPS
```

Convert between different but compatible units:

```python
from kernel_scan.core.units import Byte, Bit

# Convert between bytes and bits
byte = Byte(1.0)
bit = byte.to(Bit)  # 8.0 bits
```

### Mathematical Operations

Units support standard mathematical operations:

```python
from kernel_scan.core.units import Second, GigaFlops

# Addition
total_time = Second(1.0) + Second(2.0)  # 3.0 seconds

# Addition with different prefixes
total_perf = GigaFlops(1.5) + GigaFlops(2.5)  # 4.0 GFLOPS

# Scalar multiplication
doubled = GigaFlops(1.5) * 2  # 3.0 GFLOPS

# Division by scalar
halved = GigaFlops(3.0) / 2  # 1.5 GFLOPS
```

### Comparison

Units can be compared:

```python
from kernel_scan.core.units import GigaByte, MegaByte

# Direct comparison
gb1 = GigaByte(1.0)
gb2 = GigaByte(2.0)
is_larger = gb2 > gb1  # True

# Comparison with different prefixes
gb = GigaByte(1.0)
mb = MegaByte(500.0)  # 0.5 GB
is_smaller = mb < gb  # True
```

## Using in Performance Analysis

This module is particularly useful for GPU performance analysis, such as creating roofline models:

```python
from kernel_scan.core.units import TeraFlops, GigaBytesPerSecond, FlopsPerByte, peak_performance

# Define hardware characteristics
peak_compute = TeraFlops(15.7)  # Peak FP32 performance
peak_bandwidth = GigaBytesPerSecond(900)  # Peak memory bandwidth

# For each kernel, calculate attainable performance
kernel1_ai = FlopsPerByte(2.5)  # Arithmetic intensity of the kernel
kernel1_perf = peak_performance(peak_compute, peak_bandwidth, kernel1_ai)

# Compare against measured performance
measured_perf = TeraFlops(10.2)
efficiency = measured_perf.base_value / kernel1_perf.base_value  # As a fraction
print(f"Efficiency: {efficiency:.2%}")
```
