#!/usr/bin/env python3
"""
Test runner for Azure OpenAI and Storage connection tests.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_tests(test_type="all", verbose=False, coverage=False, markers=None):
    """
    Run Azure OpenAI and Storage tests.
    
    Args:
        test_type: Type of tests to run ('all', 'openai', 'storage', 'integration')
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        markers: Additional pytest markers to filter tests
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    
    # Select test files based on type
    if test_type == "openai":
        test_files = [test_dir / "test_azure_openai.py"]
    elif test_type == "storage":
        test_files = [test_dir / "test_azure_storage.py"]
    elif test_type == "integration":
        test_files = [test_dir / "test_azure_integration.py"]
    else:  # all
        test_files = [
            test_dir / "test_azure_openai.py",
            test_dir / "test_azure_storage.py",
            test_dir / "test_azure_integration.py"
        ]
    
    # Add test files to command
    cmd.extend([str(f) for f in test_files])
    
    # Add markers
    if markers:
        cmd.extend(["-m", markers])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return e.returncode


def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_ACCOUNT_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Warning: The following environment variables are not set:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nSome tests may be skipped or fail.")
        print("Set these variables or use mock tests only.")
        return False
    
    print("✅ All required environment variables are set.")
    return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run Azure OpenAI and Storage tests")
    parser.add_argument(
        "--type", 
        choices=["all", "openai", "storage", "integration"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    parser.add_argument(
        "--markers", "-m",
        help="Additional pytest markers to filter tests (e.g., 'azure and not slow')"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment variables and exit"
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Run only mock tests (skip tests requiring real Azure services)"
    )
    
    args = parser.parse_args()
    
    # Check environment if requested
    if args.check_env:
        check_environment()
        return 0
    
    # Add mock-only marker if requested
    if args.mock_only:
        if args.markers:
            args.markers = f"not azure and {args.markers}"
        else:
            args.markers = "not azure"
    
    # Check environment
    env_ok = check_environment()
    if not env_ok and not args.mock_only:
        print("\nUse --mock-only to run tests without Azure services.")
        return 1
    
    # Run tests
    return run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers
    )


if __name__ == "__main__":
    sys.exit(main())
