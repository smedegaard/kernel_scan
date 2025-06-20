#!/usr/bin/env python3
"""
Simple runner for the units_example.py script.

This script sets up the Python path to run the units example without
requiring installation of the kernel_scan package.
"""

import sys
from pathlib import Path


def main():
    """Run the units example with proper path setup."""
    # Get the project root directory
    project_root = Path(__file__).parent

    # Add the src directory to Python's import path
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    # Try to run the example
    try:
        # Import the example module
        from examples.units_example import main as example_main

        # Run the example's main function
        example_main()

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("\nThis could be because:")
        print("1. The project structure doesn't match expectations")
        print("2. Required dependencies are not installed")
        print("\nTry one of these solutions:")
        print("- Install the package in development mode: pip install -e .")
        print("- Check that all dependencies are installed")
        print("- Make sure you're running this script from the kernel_scan directory")

    except Exception as e:
        print(f"Error running the example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
