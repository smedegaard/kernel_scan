# Kernel Scan: Design Decisions Summary

This document outlines the key design decisions for Kernel Scan, a Python library for GPU kernel profiling focusing on AMD's Composable Kernel for GEMM operations.

## 1. Project Structure

- **Focused scope**: Initially targeting GEMM operations using AMD's Composable Kernel
- **Modern package structure**: Using the src-layout pattern for better packaging
- **Minimal dependencies**: Relying primarily on Polars and AMD's Composable Kernel

### Guidelines for the overall implementation:

1. **Dependency Direction**:
   - `core` modules should not import from `api`, `operations`, or `engines`
   - `operations` can import from `core` but not from `api` or `engines`
   - `engines` can import from `core` but not from `api` or `operations`
   - `api` can import from all other packages

2. **Type Definitions**:
   - General types (DataType, Layout, OperationType) should be in `core/types.py`
   - Operation-specific types (GemmParams) should be in `operations/gemm.py`
   - Engine-specific types should be in their respective engine modules

3. **User-Facing Classes**:
   - High-level API classes like `GemmScan` should be in the `api` package
   - These can import from and coordinate between all other packages

4. **Module Independence**:
   - Each module should be as independent as possible
   - Use dependency injection rather than direct imports where possible

```
.
├── design.md
├── examples
├── pyproject.toml
├── README.md
├── src
│   └── kernel_scan
│       ├── api                 # user-facing api
│       │   ├── gemm_scan.py    # class specifically for GEMM scan operations
│       │   └── profiler.py     # general class for profiling
│       ├── core                # core implementation
│       │   ├── config.py       # helper classes for defining configurations
│       │   ├── engine.py       # abstract base class for engines
│       │   ├── errors.py       # custom error definitions
│       │   ├── results.py      # classes for storing and handling profiling results
│       │   ├── specs.py        # classes for kernel, tensor and
│       │   └── types.py        # general type
│       ├── engines             # specific implementations of engines
│       │   ├── composable_kernel_engine.py
│       ├── operations          # utility classes for working with operations like GEMM and convolutions
│       │   ├── gemm.py
│       │   └── convolutions.py
│       └── visualization       # visualization utilities for creating plots and visualizations
│           ├── plots.py
└── uv.lock
```

## 2. Key Abstractions

- **ComputeEngine class**: Base class for compute implementations (initially just Composable Kernel)
- **KernelSpec**: Describes GEMM operations with parameters
- **ProfileResult**: Contains performance metrics from profiling
- **ProfileConfig**: Configures profiling behavior

## 3. Engine Management

- **Simplified engine design**: Initially just ComposableKernelEngine and MockEngine
- **Direct integration**: Using Composable Kernel's own profiler for performance metrics
- **Hardware detection**: Detecting AMD GPU capabilities automatically

## 4. Data Handling with Polars

- **Polars for data manipulation**: Using Polars DataFrames instead of NumPy for results processing
- **Performance focus**: Leveraging Polars' speed for analyzing large result sets
- **In-memory results**: Storing profiling results in memory as Polars DataFrames
- **Export capabilities**: Easy export to jsonl, duckdb,or other formats via Polars

## 5. Type Safety Approach

- **Python type hints**: Used throughout for better IDE support
- **Runtime validation**: Explicit validation of inputs
- **Builder pattern**: For constructing KernelSpec objects
- **Enums for constants**: For operation types, data types, and layouts

## 6. Testing Approach

- **MockEngine**: Simulated engine for testing without GPU
- **Pytest fixtures**: Common test scenarios
- **Integration tests**: Full pipeline tests with real hardware

## 7. Core Data Types

- **Operation Types**: Initially just `OperationType.GEMM`
- **Data Types**: `DataType` enum for FLOAT32, FLOAT16, etc.
- **Layout Options**: `Layout` enum for tensor memory layouts
- **GEMM Parameters**: Specialized parameters for GEMM operations

## 8. Error Handling

- **Custom exceptions**: Well-defined exception hierarchy
- **Context managers**: For resource cleanup
- **Detailed error messages**: Human-readable explanations

## 9. Configuration Approach

- **Default configurations**: Sensible defaults for most use cases
- **Builder pattern**: For complex configurations

## 10. Design Patterns

- **Factory pattern**: For creating engine instances
- **Builder pattern**: For constructing kernel specifications
- **Strategy pattern**: For different profiling strategies

## 11. Important Features Planned

- **Performance metrics**: Execution time, GFLOPS, efficiency
- **Result verification**: Verifying correctness of GEMM outputs
- **Results analysis**: Using Polars for advanced performance data analysis
- **Basic visualization**: Simple performance charts with Matplotlib

## 12. What We're Not Doing

- **No other ML frameworks**: No PyTorch, TensorFlow, or JAX integration initially
- **No operations beyond GEMM**: Starting with matrix multiplication only
- **No custom profiling tools**: Using Composable Kernel's, or other pre-existing profilers

## 14. ComposableKernelEngine Integration

The main engine will integrate with AMD's Composable Kernel library:

- Direct calls to Composable Kernel's Python bindings
- Utilizing Composable Kernel's built-in profiler for metrics collection
- Support for various GEMM configurations and data types
- Automatic hardware detection for AMD GPUs

## 15. GEMM Operations Focus

Initial focus will be exclusively on GEMM operations:

- Support for different matrix sizes and layouts
- Various data types (FP32, FP16, etc.)
- Alpha/beta scaling parameters
- Performance analysis specific to matrix multiplication

## 16. Future Expansion Plans

While initially focused, the design allows for future expansion:

- Additional operation types (convolution, element-wise, etc.)
- Support for other compute engines
- More advanced visualization and analysis tools
