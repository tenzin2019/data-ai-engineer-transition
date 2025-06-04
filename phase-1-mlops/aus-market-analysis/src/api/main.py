"""
FastAPI application for serving stock predictions and analysis.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from ..data.collector import ASXDataCollector
from ..models.trainer import StockPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ASX Stock Analysis API",
    description="API for analyzing and predicting ASX stock movements",
    version="1.0.0"
)

# Initialize components
data_collector = ASXDataCollector()
predictor = StockPredictor()

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    symbol: str
    company_name: str
    prediction_date: datetime
    up_probability: float
    down_probability: float
    confidence: float

class CompanyInfo(BaseModel):
    """Response model for company information."""
    symbol: str
    name: str
    sector: str
    market_cap: float
    pe_ratio: float
    dividend_yield: float

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to ASX Stock Analysis API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/api/v1/companies", response_model=List[CompanyInfo])
async def list_companies():
    """List all tracked companies."""
    companies = []
    for symbol in data_collector.asx_companies.keys():
        try:
            info = data_collector.get_company_info(symbol)
            companies.append(CompanyInfo(**info))
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            continue
    return companies

@app.get("/api/v1/predict/{symbol}", response_model=PredictionResponse)
async def predict_stock(symbol: str):
    """
    Get prediction for a specific stock.
    
    Args:
        symbol: ASX stock symbol
        
    Returns:
        Prediction response with probabilities
    """
    try:
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = data_collector.get_stock_data(symbol, start_date, end_date)
        
        # Train model if not already trained
        if predictor.model is None:
            predictor.train(df)
        
        # Make prediction
        prediction = predictor.predict(df)
        
        return PredictionResponse(
            symbol=symbol,
            company_name=data_collector.asx_companies.get(symbol, "Unknown"),
            prediction_date=datetime.now(),
            up_probability=prediction['up_probability'],
            down_probability=prediction['down_probability'],
            confidence=max(prediction['up_probability'], prediction['down_probability'])
        )
        
    except Exception as e:
        logger.error(f"Error making prediction for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/api/v1/analysis/{symbol}")
async def analyze_stock(symbol: str):
    """
    Get detailed analysis for a specific stock.
    
    Args:
        symbol: ASX stock symbol
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = data_collector.get_stock_data(symbol, start_date, end_date)
        
        # Calculate basic statistics
        current_price = df['Close'].iloc[-1]
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
        
        # Calculate moving averages
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        # Get RSI
        rsi = df['RSI'].iloc[-1]
        
        # Get company info
        company_info = data_collector.get_company_info(symbol)
        
        return {
            "symbol": symbol,
            "company_name": company_info['name'],
            "current_price": current_price,
            "price_change": price_change,
            "price_change_percentage": price_change_pct,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "rsi": rsi,
            "market_cap": company_info['market_cap'],
            "pe_ratio": company_info['pe_ratio'],
            "dividend_yield": company_info['dividend_yield'],
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing stock: {str(e)}"
        )

@app.get("/api/v1/market/sentiment")
async def market_sentiment():
    """
    Get overall market sentiment based on all tracked stocks.
    
    Returns:
        Dictionary with market sentiment analysis
    """
    try:
        bullish_count = 0
        bearish_count = 0
        total_stocks = len(data_collector.asx_companies)
        
        for symbol in data_collector.asx_companies.keys():
            try:
                # Get prediction for each stock
                prediction = await predict_stock(symbol)
                if prediction.up_probability > prediction.down_probability:
                    bullish_count += 1
                else:
                    bearish_count += 1
            except Exception as e:
                logger.error(f"Error getting prediction for {symbol}: {str(e)}")
                continue
        
        # Calculate sentiment
        bullish_percentage = (bullish_count / total_stocks) * 100
        bearish_percentage = (bearish_count / total_stocks) * 100
        
        return {
            "bullish_percentage": bullish_percentage,
            "bearish_percentage": bearish_percentage,
            "market_sentiment": "Bullish" if bullish_percentage > 50 else "Bearish",
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating market sentiment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating market sentiment: {str(e)}"
        ) 