#!/usr/bin/env python3
"""
Simple script to run Azure OpenAI and Storage connection tests.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run Azure connection tests."""
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("🧪 Running Azure OpenAI and Storage Connection Tests")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected. Consider using a venv.")
    
    # Check environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_STORAGE_ACCOUNT_NAME", 
        "AZURE_STORAGE_ACCOUNT_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Running mock tests only...")
        test_command = ["python", "-m", "pytest", "tests/", "-m", "not azure", "-v"]
    else:
        print("✅ All required environment variables found")
        print("Running all tests...")
        test_command = ["python", "-m", "pytest", "tests/", "-v"]
    
    print("\n" + "=" * 60)
    print("Running tests...")
    print("=" * 60)
    
    try:
        # Run tests
        result = subprocess.run(test_command, check=True)
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("\n" + "=" * 60)
        print("❌ pytest not found. Please install test dependencies:")
        print("   pip install -r requirements-test.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
