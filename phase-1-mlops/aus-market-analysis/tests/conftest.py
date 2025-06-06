"""
Common test fixtures for the ASX market analysis project.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'Open': np.random.normal(100, 5, len(dates)),
        'High': np.random.normal(105, 5, len(dates)),
        'Low': np.random.normal(95, 5, len(dates)),
        'Close': np.random.normal(100, 5, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    
    # Add technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

@pytest.fixture
def mock_company_info():
    """Create mock company information for testing."""
    return {
        'symbol': 'CBA',
        'name': 'Commonwealth Bank',
        'sector': 'Financial Services',
        'market_cap': 150000000000.0,
        'pe_ratio': 15.5,
        'dividend_yield': 4.2
    } 