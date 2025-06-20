"""
Centralized logging configuration for kernel_scan.

This module provides functions for setting up logging in a consistent
way across the entire package. This helps avoid double logging and
ensures that logging is properly configured regardless of how the
package is imported or used.
"""

import logging
import sys
from typing import Optional

# Define a default formatter
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Track if logging has been configured
_logging_configured = False


def configure_logging(level: Optional[str] = None, format_str: Optional[str] = None):
    """
    Configure the root logger for kernel_scan.

    This function should be called only once, typically when the package is first imported.
    It sets up the root logger for the package with the specified level and format.

    Args:
        level: The logging level, can be a string ('debug', 'info', etc.) or None (defaults to 'info')
        format_str: The log format string (defaults to DEFAULT_FORMAT)
    """
    global _logging_configured

    if _logging_configured:
        return

    # Convert string level to logging level
    if level is None:
        log_level = logging.INFO
    else:
        log_level = _get_level_from_string(level)

    # Set up the root logger for kernel_scan
    logger = logging.getLogger("kernel_scan")
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a handler that writes to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Create a formatter
    formatter = logging.Formatter(format_str or DEFAULT_FORMAT)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Prevent propagation to the root logger to avoid double logging
    logger.propagate = False

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This function returns a logger with the specified name, ensuring that
    it's properly prefixed with 'kernel_scan' if it's not already.

    Args:
        name: The name of the logger

    Returns:
        A logger instance
    """
    # If the name doesn't start with kernel_scan, add it
    if not name.startswith("kernel_scan"):
        name = f"kernel_scan.{name}"

    return logging.getLogger(name)


def _get_level_from_string(level: str) -> int:
    """
    Convert a string log level to a logging level constant.

    Args:
        level: The string log level ('debug', 'info', etc.)

    Returns:
        The corresponding logging level constant
    """
    level = level.lower()
    if level == "debug":
        return logging.DEBUG
    elif level == "info":
        return logging.INFO
    elif level == "warning" or level == "warn":
        return logging.WARNING
    elif level == "error":
        return logging.ERROR
    elif level == "critical":
        return logging.CRITICAL
    else:
        return logging.INFO


def update_log_level(level: str):
    """
    Update the log level of all kernel_scan loggers.

    Args:
        level: The new log level as a string ('debug', 'info', etc.)
    """
    log_level = _get_level_from_string(level)

    # Update the root kernel_scan logger
    logger = logging.getLogger("kernel_scan")
    logger.setLevel(log_level)

    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(log_level)
