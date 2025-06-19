import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class ProfileConfig:
    """
    Configuration for profiling behavior.

    Attributes:
        custom_profiler_path: Optional path to ckProfiler. If not passed, kernel_spec will try to find it automatically.
        iterations: Number of iterations to run for profiling
        warmup_iterations: Number of warmup iterations to run
        verify_results: Whether to verify results against reference implementation
        verification_tolerance: Tolerance for result verification
        engine_config: Engine-specific configuration
        output_dir: Directory for profiling outputs
        save_results: Whether to save results to disk
        result_format: Format for saving results ('csv', 'parquet', 'json')
        log_level: Logging level ('debug', 'info', 'warning', 'error')
        device_id: GPU device ID to use
        workspace_size: Workspace size in bytes
    """

    custom_profiler_path: Optional[str] = None
    iterations: int = 100
    warmup_iterations: int = 10
    verify_results: bool = True
    verification_tolerance: float = 1e-5
    engine_config: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results"
    save_results: bool = False
    result_format: str = "jsonl"
    log_level: str = "info"
    device_id: int = 0
    workspace_size: Optional[int] = None

    @classmethod
    def create_default(cls) -> "ProfileConfig":
        """Create a configuration with default values."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProfileConfig":
        """
        Create a configuration from a dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            A new ProfileConfig instance
        """
        # Create a default config
        config = cls.create_default()

        # Update with provided values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


class ProfilerConfigBuilder:
    """
    Builder pattern implementation for creating ProfileConfig objects.

    This class provides a fluent interface for constructing ProfileConfig objects
    with a chain of method calls.
    """

    def __init__(self):
        """Initialize a new ProfilerConfigBuilder with default values."""
        self._config = ProfileConfig.create_default()

    def custom_profiler_path(self, path: Optional[str]) -> "ProfilerConfigBuilder":
        """Set the custom profiler path."""
        self._config.custom_profiler_path = path
        return self

    def iterations(self, count: int) -> "ProfilerConfigBuilder":
        """Set the number of profiling iterations."""
        self._config.iterations = count
        return self

    def warmup_iterations(self, count: int) -> "ProfilerConfigBuilder":
        """Set the number of warmup iterations."""
        self._config.warmup_iterations = count
        return self

    def verify_results(self, verify: bool) -> "ProfilerConfigBuilder":
        """Set whether to verify results against reference implementation."""
        self._config.verify_results = verify
        return self

    def verification_tolerance(self, tolerance: float) -> "ProfilerConfigBuilder":
        """Set the tolerance for result verification."""
        self._config.verification_tolerance = tolerance
        return self

    def engine_config(self, **kwargs) -> "ProfilerConfigBuilder":
        """Set engine-specific configuration."""
        self._config.engine_config.update(kwargs)
        return self

    def output_dir(self, directory: str) -> "ProfilerConfigBuilder":
        """Set the directory for profiling outputs."""
        self._config.output_dir = directory
        return self

    def save_results(self, save: bool) -> "ProfilerConfigBuilder":
        """Set whether to save results to disk."""
        self._config.save_results = save
        return self

    def result_format(self, format_str: str) -> "ProfilerConfigBuilder":
        """Set the format for saving results."""
        self._config.result_format = format_str
        return self

    def log_level(self, level: str) -> "ProfilerConfigBuilder":
        """Set the logging level."""
        self._config.log_level = level
        return self

    def device_id(self, device: int) -> "ProfilerConfigBuilder":
        """Set the GPU device ID to use."""
        self._config.device_id = device
        return self

    def workspace_size(self, size: int) -> "ProfilerConfigBuilder":
        """Set the workspace size in bytes."""
        self._config.workspace_size = size
        return self

    def build(self) -> ProfileConfig:
        """Build and return a ProfileConfig object."""
        return self._config
