#!/usr/bin/env python3
"""
Health check script for Intelligent Document Analysis System.
This script verifies that all components are working correctly.
"""

import os
import sys
import time
import requests
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_environment_variables():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def check_imports():
    """Check if all required modules can be imported."""
    print("🔍 Checking Python imports...")
    
    try:
        from src.core.document_processor import DocumentProcessor
        from src.core.ai_analyzer import AIAnalyzer
        from src.database import test_connection
        from src.models import Base
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_database():
    """Check database connection."""
    print("🔍 Checking database connection...")
    
    try:
        from src.database import test_connection
        if test_connection():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_ai_services():
    """Check AI services availability."""
    print("🔍 Checking AI services...")
    
    try:
        from src.core.ai_analyzer import AIAnalyzer
        analyzer = AIAnalyzer()
        
        # Test with a simple text
        test_text = "This is a test document for health check."
        result = analyzer.analyze_document(test_text, document_type="general")
        
        if result and 'summary' in result:
            print("✅ AI services are working")
            return True
        else:
            print("❌ AI services returned unexpected result")
            return False
    except Exception as e:
        print(f"❌ AI services error: {e}")
        return False

def check_web_interface():
    """Check if web interface is accessible."""
    print("🔍 Checking web interface...")
    
    try:
        # Try to start the web interface in a subprocess
        import subprocess
        import signal
        
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "src/web/app.py", "--server.port=8501", "--server.headless=true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            # Try to access the health endpoint
            try:
                response = requests.get("http://localhost:8501/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Web interface is accessible")
                    process.terminate()
                    return True
                else:
                    print(f"❌ Web interface returned status {response.status_code}")
                    process.terminate()
                    return False
            except requests.exceptions.RequestException:
                print("❌ Web interface not accessible")
                process.terminate()
                return False
        else:
            print("❌ Web interface failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False

def check_file_permissions():
    """Check file permissions and directories."""
    print("🔍 Checking file permissions...")
    
    required_dirs = ["temp", "uploads", "logs"]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_name}")
            except Exception as e:
                print(f"❌ Failed to create directory {dir_name}: {e}")
                return False
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    return True

def main():
    """Run all health checks."""
    print("🏥 Intelligent Document Analysis System - Health Check")
    print("=" * 60)
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("Python Imports", check_imports),
        ("Database Connection", check_database),
        ("AI Services", check_ai_services),
        ("File Permissions", check_file_permissions),
        ("Web Interface", check_web_interface),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 40)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Health Check Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All health checks passed! System is ready.")
        return 0
    else:
        print("⚠️  Some health checks failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
