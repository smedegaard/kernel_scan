#!/usr/bin/env python3
"""
Run example scripts for the kernel_scan project.

This script provides an easy way to run example scripts without requiring
installation of the package. It handles Python path setup automatically.

Usage:
    python run_example.py units_example
    python run_example.py --list
"""

import argparse
import importlib
import os
import sys
from pathlib import Path


def list_examples():
    """List all available example scripts."""
    examples_dir = Path(__file__).parent / "examples"

    if not examples_dir.exists():
        print("Error: Examples directory not found.")
        sys.exit(1)

    print("Available examples:")
    print("-" * 30)

    for file in sorted(examples_dir.glob("*.py")):
        if file.name == "__init__.py":
            continue

        name = file.stem
        # Try to extract the docstring for a description
        try:
            with open(file, "r") as f:
                content = f.read()
                docstring = content.split('"""')[1].split('"""')[0].strip()
                description = docstring.split("\n")[0]  # First line only
        except (IndexError, FileNotFoundError):
            description = "No description available"

        print(f"{name:20} - {description}")

    print("\nRun an example with: python run_example.py <example_name>")


def setup_environment():
    """Add the src directory to the Python path to make imports work."""
    project_root = Path(__file__).parent
    src_path = project_root / "src"

    if not src_path.exists():
        print(f"Error: Source directory not found at {src_path}")
        sys.exit(1)

    sys.path.insert(0, str(src_path))

    # Also add the project root to allow imports from examples directory
    sys.path.insert(0, str(project_root))


def run_example(example_name):
    """Run the specified example script."""
    setup_environment()

    examples_dir = Path(__file__).parent / "examples"
    example_path = examples_dir / f"{example_name}.py"

    if not example_path.exists():
        print(f"Error: Example '{example_name}' not found.")
        print("Use --list to see available examples.")
        sys.exit(1)

    print(f"Running example: {example_name}")
    print("-" * 50)

    # Try to import and run the example as a module
    try:
        example_module = importlib.import_module(f"examples.{example_name}")
        if hasattr(example_module, "main"):
            example_module.main()
        else:
            print("Note: Example has no main() function, imported as module only.")
    except ImportError as e:
        # Fall back to executing the file directly
        print(f"Note: Could not import as module ({e}), executing file directly.")
        example_dir = os.path.dirname(example_path)
        current_dir = os.getcwd()

        try:
            os.chdir(example_dir)
            with open(example_path) as f:
                exec(f.read(), {"__name__": "__main__"})
        finally:
            os.chdir(current_dir)


def main():
    """Parse command line arguments and run the selected example."""
    parser = argparse.ArgumentParser(
        description="Run kernel_scan examples without installation."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "example", nargs="?", help="Name of the example to run (without .py extension)"
    )
    group.add_argument(
        "--list", "-l", action="store_true", help="List all available examples"
    )

    args = parser.parse_args()

    if args.list:
        list_examples()
    elif args.example:
        run_example(args.example)


if __name__ == "__main__":
    main()
