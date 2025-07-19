"""
Monitoring and Observability for Financial Behavior Insights

This module provides comprehensive monitoring, logging, and observability features
following MLOps best practices for production ML systems.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from .config import get_config, MODEL_CONFIG, DEPLOYMENT_CONFIG

@dataclass
class PredictionRecord:
    """Record for tracking individual predictions."""
    timestamp: str
    input_data: Dict[str, Any]
    prediction: Any
    probability: Optional[float] = None
    model_version: Optional[str] = None
    endpoint_name: Optional[str] = None
    response_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    prediction_count: int
    error_count: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

class ModelMonitor:
    """Comprehensive model monitoring and observability."""
    
    def __init__(self, config=None):
        """Initialize the model monitor."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Setup monitoring directories
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)
        
        self.predictions_file = self.monitoring_dir / "predictions.jsonl"
        self.metrics_file = self.monitoring_dir / "metrics.json"
        self.alerts_file = self.monitoring_dir / "alerts.json"
        
        # Initialize metrics
        self.current_metrics = ModelMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0, auc_score=0.0,
            prediction_count=0, error_count=0, avg_response_time_ms=0.0,
            p95_response_time_ms=0.0, p99_response_time_ms=0.0
        )
        
        # Load historical metrics
        self.load_metrics()
    
    def log_prediction(self, prediction_record: PredictionRecord):
        """Log a prediction record."""
        try:
            # Add timestamp if not provided
            if not prediction_record.timestamp:
                prediction_record.timestamp = datetime.now().isoformat()
            
            # Add model version and endpoint if not provided
            if not prediction_record.model_version:
                prediction_record.model_version = "latest"
            if not prediction_record.endpoint_name:
                prediction_record.endpoint_name = MODEL_CONFIG.endpoint_name
            
            # Write to JSONL file
            with open(self.predictions_file, 'a') as f:
                f.write(json.dumps(asdict(prediction_record)) + '\n')
            
            # Update metrics
            self._update_metrics(prediction_record)
            
            # Check for anomalies
            self._check_anomalies(prediction_record)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log prediction: {e}")
    
    def _update_metrics(self, prediction_record: PredictionRecord):
        """Update current metrics with new prediction."""
        # Update counts
        self.current_metrics.prediction_count += 1
        if not prediction_record.success:
            self.current_metrics.error_count += 1
        
        # Update response time metrics
        if prediction_record.response_time_ms:
            # Simple moving average for response time
            if self.current_metrics.avg_response_time_ms == 0:
                self.current_metrics.avg_response_time_ms = prediction_record.response_time_ms
            else:
                alpha = 0.1  # Smoothing factor
                self.current_metrics.avg_response_time_ms = (
                    alpha * prediction_record.response_time_ms +
                    (1 - alpha) * self.current_metrics.avg_response_time_ms
                )
    
    def _check_anomalies(self, prediction_record: PredictionRecord):
        """Check for anomalies and trigger alerts."""
        alerts = []
        
        # Check response time anomalies
        if prediction_record.response_time_ms:
            if prediction_record.response_time_ms > 5000:  # 5 seconds
                alerts.append({
                    "type": "high_response_time",
                    "severity": "warning",
                    "message": f"High response time: {prediction_record.response_time_ms}ms",
                    "timestamp": prediction_record.timestamp
                })
        
        # Check error rate
        error_rate = self.current_metrics.error_count / max(self.current_metrics.prediction_count, 1)
        if error_rate > 0.05:  # 5% error rate
            alerts.append({
                "type": "high_error_rate",
                "severity": "critical",
                "message": f"High error rate: {error_rate:.2%}",
                "timestamp": prediction_record.timestamp
            })
        
        # Log alerts
        if alerts:
            self._log_alerts(alerts)
    
    def _log_alerts(self, alerts: List[Dict[str, Any]]):
        """Log alerts to file."""
        try:
            existing_alerts = []
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r') as f:
                    existing_alerts = json.load(f)
            
            existing_alerts.extend(alerts)
            
            # Keep only last 1000 alerts
            if len(existing_alerts) > 1000:
                existing_alerts = existing_alerts[-1000:]
            
            with open(self.alerts_file, 'w') as f:
                json.dump(existing_alerts, f, indent=2)
            
            # Log to console
            for alert in alerts:
                self.logger.warning(f"🚨 ALERT: {alert['message']}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log alerts: {e}")
    
    def get_metrics(self, window_hours: int = 24) -> ModelMetrics:
        """Get metrics for the specified time window."""
        try:
            # Load predictions from the time window
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            predictions = []
            
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    for line in f:
                        try:
                            pred_data = json.loads(line.strip())
                            pred_time = datetime.fromisoformat(pred_data['timestamp'])
                            if pred_time >= cutoff_time:
                                predictions.append(pred_data)
                        except:
                            continue
            
            # Calculate metrics
            if predictions:
                return self._calculate_metrics(predictions)
            else:
                return self.current_metrics
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get metrics: {e}")
            return self.current_metrics
    
    def _calculate_metrics(self, predictions: List[Dict[str, Any]]) -> ModelMetrics:
        """Calculate metrics from prediction data."""
        if not predictions:
            return self.current_metrics
        
        # Extract response times
        response_times = [p.get('response_time_ms', 0) for p in predictions if p.get('response_time_ms')]
        
        # Calculate response time percentiles
        if response_times:
            response_times = np.array(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            avg_response_time = np.mean(response_times)
        else:
            p95_response_time = p99_response_time = avg_response_time = 0.0
        
        # Calculate success rate
        success_count = sum(1 for p in predictions if p.get('success', True))
        error_count = len(predictions) - success_count
        
        return ModelMetrics(
            accuracy=0.0,  # Would need ground truth for accuracy
            precision=0.0,  # Would need ground truth for precision
            recall=0.0,     # Would need ground truth for recall
            f1_score=0.0,   # Would need ground truth for F1
            auc_score=0.0,  # Would need ground truth for AUC
            prediction_count=len(predictions),
            error_count=error_count,
            avg_response_time_ms=float(avg_response_time),
            p95_response_time_ms=float(p95_response_time),
            p99_response_time_ms=float(p99_response_time)
        )
    
    def save_metrics(self):
        """Save current metrics to file."""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": asdict(self.current_metrics)
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    metrics_dict = data.get('metrics', {})
                    
                    # Update current metrics
                    for key, value in metrics_dict.items():
                        if hasattr(self.current_metrics, key):
                            setattr(self.current_metrics, key, value)
                            
        except Exception as e:
            self.logger.error(f"❌ Failed to load metrics: {e}")
    
    def generate_report(self, window_hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        try:
            metrics = self.get_metrics(window_hours)
            
            # Load recent alerts
            alerts = []
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r') as f:
                    all_alerts = json.load(f)
                    cutoff_time = datetime.now() - timedelta(hours=window_hours)
                    alerts = [
                        alert for alert in all_alerts
                        if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
                    ]
            
            # Calculate success rate
            success_rate = 1.0 - (metrics.error_count / max(metrics.prediction_count, 1))
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "window_hours": window_hours,
                "summary": {
                    "total_predictions": metrics.prediction_count,
                    "success_rate": success_rate,
                    "error_rate": 1.0 - success_rate,
                    "avg_response_time_ms": metrics.avg_response_time_ms,
                    "p95_response_time_ms": metrics.p95_response_time_ms,
                    "p99_response_time_ms": metrics.p99_response_time_ms
                },
                "alerts": {
                    "total_alerts": len(alerts),
                    "critical_alerts": len([a for a in alerts if a['severity'] == 'critical']),
                    "warning_alerts": len([a for a in alerts if a['severity'] == 'warning']),
                    "recent_alerts": alerts[-10:]  # Last 10 alerts
                },
                "recommendations": self._generate_recommendations(metrics, alerts)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, metrics: ModelMetrics, alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on metrics and alerts."""
        recommendations = []
        
        # Response time recommendations
        if metrics.avg_response_time_ms > 1000:
            recommendations.append("Consider optimizing model inference or scaling up resources")
        
        if metrics.p95_response_time_ms > 5000:
            recommendations.append("High p95 response time detected - investigate performance bottlenecks")
        
        # Error rate recommendations
        error_rate = metrics.error_count / max(metrics.prediction_count, 1)
        if error_rate > 0.05:
            recommendations.append("High error rate detected - investigate model performance and data quality")
        
        # Alert-based recommendations
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        if critical_alerts:
            recommendations.append("Critical alerts detected - immediate attention required")
        
        if not recommendations:
            recommendations.append("System performing within normal parameters")
        
        return recommendations

class HealthChecker:
    """Health checking for the deployed model."""
    
    def __init__(self, config=None):
        """Initialize the health checker."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def check_endpoint_health(self) -> Dict[str, Any]:
        """Check the health of the deployed endpoint."""
        try:
            endpoint_url = self.config.get_endpoint_url()
            
            # Test basic connectivity
            start_time = time.time()
            response = requests.get(endpoint_url.replace('/score', '/health'), timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "endpoint_url": endpoint_url,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_code": response.status_code,
                "response_time_ms": response_time,
                "details": {}
            }
            
            # Check response time
            if response_time > 5000:
                health_status["details"]["slow_response"] = f"Response time: {response_time:.2f}ms"
            
            # Check if endpoint is accessible
            if response.status_code != 200:
                health_status["details"]["accessibility"] = f"Endpoint returned {response.status_code}"
            
            return health_status
            
        except requests.exceptions.Timeout:
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint_url": endpoint_url,
                "status": "unhealthy",
                "response_code": None,
                "response_time_ms": None,
                "details": {"timeout": "Request timed out"}
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint_url": endpoint_url,
                "status": "unhealthy",
                "response_code": None,
                "response_time_ms": None,
                "details": {"error": str(e)}
            }
    
    def check_model_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Check model performance with test data."""
        try:
            endpoint_url = self.config.get_endpoint_url()
            
            # Get endpoint key
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.config.azure.subscription_id,
                resource_group=self.config.azure.resource_group,
                workspace_name=self.config.azure.workspace_name
            )
            
            keys = ml_client.online_endpoints.get_keys(MODEL_CONFIG.endpoint_name)
            # Extract the primary key properly
            if hasattr(keys, 'primary_key'):
                endpoint_key = keys.primary_key
            elif hasattr(keys, 'key1'):
                endpoint_key = keys.key1
            else:
                # Fallback: try to get the key using Azure CLI
                import subprocess
                result = subprocess.run([
                    'az', 'ml', 'online-endpoint', 'get-credentials',
                    '--name', MODEL_CONFIG.endpoint_name,
                    '--resource-group', self.config.azure.resource_group,
                    '--workspace-name', self.config.azure.workspace_name,
                    '--query', 'primaryKey',
                    '-o', 'tsv'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    endpoint_key = result.stdout.strip()
                else:
                    raise ValueError("Could not get endpoint key")
            
            # Test predictions
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {endpoint_key}'
            }
            
            performance_results = {
                "timestamp": datetime.now().isoformat(),
                "test_samples": len(test_data),
                "predictions": [],
                "response_times": [],
                "errors": []
            }
            
            for idx, row in test_data.head(10).iterrows():  # Test with first 10 samples
                try:
                    payload = row.to_dict()
                    
                    start_time = time.time()
                    response = requests.post(
                        endpoint_url,
                        json=payload,
                        headers=headers,
                        timeout=30
                    )
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        result = response.json()
                        performance_results["predictions"].append(result)
                        performance_results["response_times"].append(response_time)
                    else:
                        performance_results["errors"].append({
                            "sample": idx,
                            "status_code": response.status_code,
                            "error": response.text
                        })
                        
                except Exception as e:
                    performance_results["errors"].append({
                        "sample": idx,
                        "error": str(e)
                    })
            
            # Calculate performance metrics
            if performance_results["response_times"]:
                performance_results["avg_response_time_ms"] = np.mean(performance_results["response_times"])
                performance_results["p95_response_time_ms"] = np.percentile(performance_results["response_times"], 95)
                performance_results["success_rate"] = len(performance_results["predictions"]) / len(test_data.head(10))
            else:
                performance_results["avg_response_time_ms"] = 0
                performance_results["p95_response_time_ms"] = 0
                performance_results["success_rate"] = 0
            
            return performance_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to check model performance: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

# Global instances
model_monitor = ModelMonitor()
health_checker = HealthChecker()

def log_prediction(input_data: Dict[str, Any], prediction: Any, 
                  probability: Optional[float] = None, 
                  response_time_ms: Optional[float] = None,
                  success: bool = True, error_message: Optional[str] = None):
    """Convenience function to log a prediction."""
    prediction_record = PredictionRecord(
        timestamp=datetime.now().isoformat(),
        input_data=input_data,
        prediction=prediction,
        probability=probability,
        response_time_ms=response_time_ms,
        success=success,
        error_message=error_message
    )
    model_monitor.log_prediction(prediction_record)

def get_monitoring_report(window_hours: int = 24) -> Dict[str, Any]:
    """Get a comprehensive monitoring report."""
    return model_monitor.generate_report(window_hours)

def check_system_health() -> Dict[str, Any]:
    """Check overall system health."""
    return health_checker.check_endpoint_health() 