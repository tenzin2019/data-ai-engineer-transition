"""
Health check endpoint for Azure App Service.
This module provides a simple health check endpoint that Azure can use to monitor the application.
"""

import os
import json
import psutil
from datetime import datetime
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status of the application.
    
    Returns:
        Dict containing health status information
    """
    try:
        # Basic system information
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        # Application information
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "port": os.getenv("PORT", "8000"),
            "system": {
                "memory": {
                    "total": memory_info.total,
                    "available": memory_info.available,
                    "percent": memory_info.percent,
                    "used": memory_info.used
                },
                "disk": {
                    "total": disk_info.total,
                    "free": disk_info.free,
                    "used": disk_info.used,
                    "percent": (disk_info.used / disk_info.total) * 100
                },
                "cpu_percent": psutil.cpu_percent(interval=1)
            },
            "azure": {
                "app_service": True,
                "container": True,
                "region": os.getenv("WEBSITE_SITE_NAME", "unknown"),
                "instance_id": os.getenv("WEBSITE_INSTANCE_ID", "unknown")
            },
            "services": {
                "azure_openai": check_azure_openai_health(),
                "azure_storage": check_azure_storage_health(),
                "database": check_database_health()
            }
        }
        
        # Determine overall health status
        if memory_info.percent > 90:
            health_data["status"] = "warning"
            health_data["warnings"] = ["High memory usage"]
        
        if disk_info.percent > 90:
            health_data["status"] = "warning"
            if "warnings" not in health_data:
                health_data["warnings"] = []
            health_data["warnings"].append("High disk usage")
        
        # Check if any critical services are down
        critical_services_down = []
        for service, status in health_data["services"].items():
            if not status:
                critical_services_down.append(service)
        
        if critical_services_down:
            health_data["status"] = "unhealthy"
            health_data["errors"] = [f"Critical services down: {', '.join(critical_services_down)}"]
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

def check_azure_openai_health() -> bool:
    """Check if Azure OpenAI service is accessible."""
    try:
        # Check if required environment variables are set
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if not endpoint or not api_key:
            return False
        
        # In a real implementation, you would make a test API call
        # For now, just check if the configuration is present
        return endpoint.startswith("https://") and len(api_key) > 10
        
    except Exception:
        return False

def check_azure_storage_health() -> bool:
    """Check if Azure Storage service is accessible."""
    try:
        # Check if required environment variables are set
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        
        if not account_name or not account_key:
            return False
        
        # In a real implementation, you would make a test API call
        # For now, just check if the configuration is present
        return len(account_name) > 0 and len(account_key) > 10
        
    except Exception:
        return False

def check_database_health() -> bool:
    """Check if database is accessible."""
    try:
        # Check if database URL is configured
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            return False
        
        # In a real implementation, you would test the database connection
        # For now, just check if the configuration is present
        return database_url.startswith("postgresql://")
        
    except Exception:
        return False

def create_health_response(status_code: int = 200) -> tuple:
    """
    Create a health check response.
    
    Args:
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_data, status_code)
    """
    health_data = get_health_status()
    
    # Determine HTTP status code based on health status
    if health_data["status"] == "healthy":
        http_status = 200
    elif health_data["status"] == "warning":
        http_status = 200  # Still return 200 but with warning
    else:  # unhealthy
        http_status = 503
    
    return health_data, http_status

# Simple Flask app for health checks
def create_health_app():
    """Create a simple Flask app for health checks."""
    try:
        from flask import Flask, jsonify
        app = Flask(__name__)
        
        @app.route('/health')
        def health():
            """Health check endpoint."""
            health_data, status_code = create_health_response()
            return jsonify(health_data), status_code
        
        @app.route('/health/ready')
        def readiness():
            """Readiness probe endpoint."""
            health_data, status_code = create_health_response()
            if health_data["status"] in ["healthy", "warning"]:
                return jsonify({"status": "ready"}), 200
            else:
                return jsonify({"status": "not ready"}), 503
        
        @app.route('/health/live')
        def liveness():
            """Liveness probe endpoint."""
            return jsonify({"status": "alive"}), 200
        
        return app
        
    except ImportError:
        # If Flask is not available, return a simple function
        def simple_health():
            health_data, status_code = create_health_response()
            return json.dumps(health_data), status_code
        return simple_health

if __name__ == "__main__":
    # Run health check server
    app = create_health_app()
    if hasattr(app, 'run'):
        port = int(os.getenv("PORT", 8000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Simple health check
        health_data, status_code = create_health_response()
        print(f"Status: {status_code}")
        print(json.dumps(health_data, indent=2))
