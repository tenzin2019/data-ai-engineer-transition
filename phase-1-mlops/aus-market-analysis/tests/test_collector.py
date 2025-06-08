"""
Tests for the ASX data collector module.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from src.data.collector import ASXDataCollector

def test_init():
    """Test ASXDataCollector initialization."""
    collector = ASXDataCollector()
    assert isinstance(collector.asx_companies, dict)
    assert len(collector.asx_companies) > 0
    assert 'CBA' in collector.asx_companies

@patch('yfinance.Ticker')
def test_get_stock_data(mock_ticker, sample_stock_data):
    """Test stock data collection."""
    # Setup mock
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_stock_data
    mock_ticker.return_value = mock_ticker_instance
    
    collector = ASXDataCollector()
    df = collector.get_stock_data('CBA')
    
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    assert all(col in df.columns for col in ['SMA_20', 'SMA_50', 'RSI'])
    assert not df.empty

@patch('requests.get')
def test_get_company_info(mock_get, mock_company_info):
    """Test company information collection."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = '<html><body>Mock ASX page</body></html>'
    mock_get.return_value = mock_response
    
    collector = ASXDataCollector()
    info = collector.get_company_info('CBA')
    
    assert isinstance(info, dict)
    assert 'symbol' in info
    assert 'name' in info
    assert info['symbol'] == 'CBA'

def test_calculate_rsi():
    """Test RSI calculation."""
    collector = ASXDataCollector()
    prices = pd.Series([100, 102, 101, 103, 104, 102, 101, 100, 99, 98])
    rsi = collector._calculate_rsi(prices, period=2)
    
    assert isinstance(rsi, pd.Series)
    assert not rsi.isna().all()
    assert all(0 <= x <= 100 for x in rsi.dropna())

@patch('yfinance.Ticker')
def test_get_all_companies_data(mock_ticker, sample_stock_data):
    """Test collection of data for all companies."""
    # Setup mock
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_stock_data
    mock_ticker.return_value = mock_ticker_instance
    
    collector = ASXDataCollector()
    data = collector.get_all_companies_data()
    
    assert isinstance(data, dict)
    assert len(data) == len(collector.asx_companies)
    assert all(isinstance(df, pd.DataFrame) for df in data.values())

def test_get_stock_data_invalid_symbol():
    """Test handling of invalid stock symbol."""
    collector = ASXDataCollector()
    with pytest.raises(Exception):
        collector.get_stock_data('INVALID')

def test_get_company_info_invalid_symbol():
    """Test handling of invalid company symbol."""
    collector = ASXDataCollector()
    with pytest.raises(Exception):
        collector.get_company_info('INVALID') 