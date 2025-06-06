"""
Monitoring utilities for the ASX market analysis project.
"""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define metrics
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['symbol']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests',
    ['symbol']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy for predictions',
    ['symbol']
)

API_REQUESTS = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method']
)

def track_prediction_metrics(symbol: str):
    """
    Decorator to track prediction metrics.
    
    Args:
        symbol: Stock symbol being predicted
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            PREDICTION_REQUESTS.labels(symbol=symbol).inc()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                # Update accuracy if available
                if hasattr(result, 'confidence'):
                    MODEL_ACCURACY.labels(symbol=symbol).set(result.confidence)
                return result
            finally:
                PREDICTION_LATENCY.labels(symbol=symbol).observe(time.time() - start_time)
        return wrapper
    return decorator

def track_api_metrics(endpoint: str, method: str):
    """
    Decorator to track API metrics.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            API_REQUESTS.labels(endpoint=endpoint, method=method).inc()
            return func(*args, **kwargs)
        return wrapper
    return decorator 