# Kernel Scan: GPU Kernel Profiling Made Simple

A Python library for profiling GPU compute kernels across different hardware and library backends, focused on simplicity and usability.

## Purpose

Kernel Scan simplifies the complex task of GPU kernel profiling by providing a clean, Pythonic interface to various compute libraries. The library allows data scientists, ML engineers, and performance specialists to easily benchmark and compare different GPU kernel implementations across hardware platforms.

## Features

[x] Object-oriented design with multiple compute engine backends
[x] Clean, ergonomic units system for performance metrics
[] Type validation through Python type hints
[] Dynamic backend discovery and loading
[] Hardware-agnostic testing through mock engines
[x] Comprehensive metrics collection and visualization

## Philosophy

Kernel Scan embraces Python's simplicity while maintaining the rigor needed for performance analysis. THe goal is to make GPU profiling:

- **Accessible**: No need for complex C/C++ extensions or custom builds
- **Intuitive**: Follows Python conventions and integrates with familiar tools
- **Informative**: Provides clear, actionable insights without overwhelming details
- **Reliable**: Consistent results you can trust across platforms

## Installation

Kernel Scan uses optional dependencies to control which compute engines are included. The base package includes only core functionality and mock engines for testing.

```bash
# Base installation (no actual compute engines)
uv add kernel-scan

# Install with specific compute library support
uv add kernel-scan[composable_kernel]     # AMD COMPOSABLE_KERNEL
```

## Quick Start

```python
from kernel_scan import (
    KernelSpec,
    Profiler,
    EngineType,
    OperationType,
    DataType,
    Layout,
    GemmParams,
    TensorSpec
)

# Create a GEMM kernel specification
kernel_spec = (
    KernelSpec.builder()
    .operation_type(OperationType.GEMM)
    .data_type(DataType.FLOAT32)
    .operation_params(GemmParams(
        m=1024, n=1024, k=1024,
        alpha=1.0, beta=0.0,
        layout_a=Layout.ROW_MAJOR,
        layout_b=Layout.ROW_MAJOR,
        layout_c=Layout.ROW_MAJOR
    ))
    .inputs(
        a=TensorSpec.create_2d(1024, 1024, Layout.ROW_MAJOR, DataType.FLOAT32),
        b=TensorSpec.create_2d(1024, 1024, Layout.ROW_MAJOR, DataType.FLOAT32)
    )
    .outputs(
        c=TensorSpec.create_2d(1024, 1024, Layout.ROW_MAJOR, DataType.FLOAT32)
    )
    .iterations(100)
    .build()
)

# Profile with a specific engine
profiler = Profiler()
result = profiler.profile_with_engine(
    kernel_spec,
    EngineType.COMPOSABLE_KERNEL
    warmup_iterations=10
)

print(f"Execution time: {result.timing.avg_kernel_time_ms:.3f} ms")
print(f"Throughput: {result.metrics['gflops']:.2f} GFLOPS")

# Create visualization
result.plot_performance()
```

## Units System

Kernel Scan includes a clean, ergonomic units system for working with performance metrics:

```python
from kernel_scan.core.units import TeraFlops, GigaBytesPerSecond, FlopsPerByte, peak_performance

# Create performance metrics with explicit prefixes
compute = TeraFlops(15.7)          # 15.7 TFLOPS
bandwidth = GigaBytesPerSecond(900)  # 900 GB/s

# Convert between units with different prefixes
gflops = compute.to_giga()  # 15700 GFLOPS
print(gflops)  # "15700.0 GFLOPS"

# Calculate arithmetic intensity
intensity = FlopsPerByte(4.0)  # 4.0 FLOPS/Byte

# Apply the roofline model
attainable = peak_performance(compute, bandwidth, intensity)
print(f"Peak attainable performance: {attainable}")
```

The units system handles:
- Automatic conversions between unit prefixes (kilo, mega, giga, tera, etc.)
- Dimension tracking to prevent invalid operations
- Mathematical operations and comparisons between compatible units
- Performance analysis calculations like arithmetic intensity and roofline modeling
