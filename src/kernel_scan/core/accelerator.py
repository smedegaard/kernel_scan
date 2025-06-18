"""
Hardware specifications for accelerator profiling.

This module provides classes and functions for representing
hardware specifications used in profiling and analysis.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional

from kernel_scan.core.types import DataType

log = logging.getLogger(__name__)

# GPU specifications map organized by vendor and model
GPU_SPECS = {
    "AMD": {
        "Radeon RX 7900 XTX": {
            "name": "Radeon RX 7900 XTX",
            "peak_memory_bandwidth_gbps": 960.0,
            "peak_performance": {
                DataType.FLOAT32: 61.4,  # FP32 TFLOPS
                DataType.FLOAT16: 122.8,  # FP16 TFLOPS (2x FP32)
                DataType.INT8: 245.6,  # INT8 TOPS (4x FP32)
            },
            "memory_size_gb": 24.0,
            "additional_specs": {
                "stream_processors": 12288,
                "compute_units": 96,
                "memory_type": "GDDR6",
                "memory_bus_width": 384,
                "architecture": "RDNA 3",
                "release_year": 2022,
            },
        },
        "Instinct MI250X": {
            "name": "AMD Instinct MI250X",
            "peak_memory_bandwidth_gbps": 3200.0,
            "peak_performance": {
                DataType.FLOAT64: 47.9,  # FP64 TFLOPS
                DataType.FLOAT32: 95.7,  # FP32 TFLOPS
                DataType.FLOAT16: 383.0,  # FP16 TFLOPS
                DataType.BFLOAT16: 383.0,  # BF16 TFLOPS
                DataType.INT8: 766.0,  # INT8 TOPS
            },
            "memory_size_gb": 128.0,
            "additional_specs": {
                "stream_processors": 14080,
                "compute_units": 220,
                "memory_type": "HBM2e",
                "memory_bus_width": 8192,
                "architecture": "CDNA 2",
                "release_year": 2021,
            },
        },
        "Instinct MI300X": {
            "name": "AMD Instinct MI300X",
            "peak_memory_bandwidth_gbps": 5300.0,  # 5.3 TB/s
            "peak_performance": {
                DataType.FLOAT64: 152.8,  # FP64 TFLOPS
                DataType.FLOAT32: 305.6,  # FP32 TFLOPS
                DataType.FLOAT16: 611.2,  # FP16 TFLOPS
                DataType.BFLOAT16: 611.2,  # BF16 TFLOPS
                DataType.INT8: 1222.4,  # INT8 TOPS
            },
            "memory_size_gb": 192.0,
            "additional_specs": {
                "stream_processors": 304 * 256,  # 304 CUs * 256 SPs per CU
                "compute_units": 304,
                "memory_type": "HBM3",
                "memory_bus_width": 8192,
                "architecture": "CDNA 3",
                "release_year": 2023,
                "chiplets": 8,  # 8 GCD chiplets
                "matrix_engines": 304,  # One per CU
            },
        },
    },
    "NVIDIA": {
        "A100": {
            "name": "NVIDIA A100",
            "peak_memory_bandwidth_gbps": 2039.0,
            "peak_performance": {
                DataType.FLOAT64: 9.7,  # FP64 TFLOPS
                DataType.FLOAT32: 19.5,  # FP32 TFLOPS
                DataType.FLOAT16: 312.0,  # FP16 TFLOPS (with Tensor Cores)
                DataType.BFLOAT16: 312.0,  # BF16 TFLOPS (with Tensor Cores)
                DataType.INT8: 624.0,  # INT8 TOPS (with Tensor Cores)
                DataType.INT4: 1248.0,  # INT4 TOPS (with Tensor Cores)
            },
            "memory_size_gb": 80.0,
            "additional_specs": {
                "cuda_cores": 6912,
                "tensor_cores": 432,
                "memory_type": "HBM2e",
                "memory_bus_width": 5120,
                "architecture": "Ampere",
                "release_year": 2020,
            },
        },
        "H100": {
            "name": "NVIDIA H100",
            "peak_memory_bandwidth_gbps": 3350.0,
            "peak_performance": {
                DataType.FLOAT64: 67.0,  # FP64 TFLOPS (with Tensor Cores)
                DataType.FLOAT32: 67.0,  # FP32 TFLOPS
                DataType.FLOAT16: 989.0,  # FP16 TFLOPS (with Tensor Cores)
                DataType.BFLOAT16: 989.0,  # BF16 TFLOPS (with Tensor Cores)
                DataType.INT8: 1979.0,  # INT8 TOPS (with Tensor Cores)
            },
            "memory_size_gb": 80.0,
            "additional_specs": {
                "cuda_cores": 16896,
                "tensor_cores": 528,
                "memory_type": "HBM3",
                "memory_bus_width": 5120,
                "architecture": "Hopper",
                "release_year": 2022,
            },
        },
    },
}


@dataclass
class AcceleratorSpec:
    """
    Accelerator hardware specifications for performance analysis.

    Attributes:
        name: Name or model of the accelerator
        peak_memory_bandwidth_gbps: Peak memory bandwidth in GB/s
        peak_performance: Dict mapping data types to TFLOPS values
        memory_size_gb: Size of accelerator memory in GB
        additional_specs: Additional hardware specifications
    """

    name: str = "Unknown"
    peak_memory_bandwidth_gbps: Optional[float] = None
    peak_performance: Dict[DataType, float] = field(default_factory=dict)
    memory_size_gb: Optional[float] = None
    additional_specs: Dict[str, any] = field(default_factory=dict)

    @property
    def peak_bandwidth(self) -> float:
        """Return peak memory bandwidth in GB/s."""
        return (
            self.peak_memory_bandwidth_gbps
            if self.peak_memory_bandwidth_gbps is not None
            else 0.0
        )

    def get_peak_compute(self, precision: DataType = DataType.FLOAT32) -> float:
        """
        Get peak compute performance for the specified precision.

        Args:
            precision: Precision format to query

        Returns:
            Peak compute performance in TFLOPS
        """

        if precision in self.peak_performance:
            return self.peak_performance[precision]
        else:
            log.warning(
                f"No peak performance data for {precision.name} on {self.name}. Using FLOAT32 as fallback."
            )
            return self.peak_performance.get(DataType.FLOAT32, 0.0)

    @staticmethod
    def detect_hardware() -> "AcceleratorSpec":
        """
        Detect GPU hardware information and create an AcceleratorSpec instance.

        This method attempts to identify the GPU hardware by querying system tools
        and creates an AcceleratorSpec instance with the detected values.

        Returns:
            AcceleratorSpec instance with detected hardware information
        """
        # Start with unknown values
        specs = AcceleratorSpec()

        # Try to detect the GPU from system
        detected_vendor = None
        detected_model = None

        # First try AMD GPUs with rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--showallinfo", "--json"],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                try:
                    rocm_info = json.loads(result.stdout)
                    if isinstance(rocm_info, dict):
                        for card_id, value in rocm_info.items():
                            # Find the primary GPU - we'll use the first discrete GPU
                            device_name = rocm_info[card_id]["Device Name"]
                            # Check if this is a known AMD GPU
                            detected_vendor = "AMD"
                            log.info(f"Detected AMD GPU: {device_name}")

                            # Look for exact match
                            if device_name in GPU_SPECS["AMD"]:
                                detected_model = device_name
                                break

                            # If we found a match, no need to check other cards
                            if detected_model:
                                break
                except (json.JSONDecodeError, KeyError) as e:
                    log.info(f"Failed to parse rocm-smi output: {e}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.info(f"Failed to run rocm-smi: {e}")

        # todo: If AMD detection failed, try NVIDIA GPUs

        # If we detected a known GPU, update the AcceleratorSpec
        if detected_vendor and detected_model:
            log.info(
                f"Found matching GPU model: {detected_model} from vendor {detected_vendor}"
            )
            gpu_specs = GPU_SPECS[detected_vendor][detected_model]

            # Create AcceleratorSpec with detected values
            specs = AcceleratorSpec(
                name=gpu_specs["name"],
                peak_memory_bandwidth_gbps=gpu_specs["peak_memory_bandwidth_gbps"],
                peak_performance=gpu_specs["peak_performance"].copy(),
                memory_size_gb=gpu_specs["memory_size_gb"],
                additional_specs=gpu_specs["additional_specs"].copy(),
            )

            # Add vendor information
            specs.additional_specs["vendor"] = detected_vendor
        else:
            # If we couldn't find a match in our database, log a warning
            if detected_vendor:
                log.warning(
                    f"Detected {detected_vendor} GPU, but couldn't match it to a known model in the database."
                )
                specs.additional_specs["vendor"] = detected_vendor
            else:
                log.warning("Could not detect GPU vendor or model.")

        return specs
