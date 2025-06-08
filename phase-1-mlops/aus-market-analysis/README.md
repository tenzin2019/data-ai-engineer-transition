# Australian Market Analysis & Prediction System

## Overview
This project provides a comprehensive analysis and prediction system for top Australian companies listed on the ASX. It combines machine learning, data engineering, and MLOps practices to deliver actionable insights and predictions.

## Features
- Real-time data collection from ASX
- Analysis of top 50 ASX companies
- Stock price prediction models
- Market sentiment analysis
- API endpoints for real-time analysis
- Automated model retraining pipeline
- Performance monitoring dashboard

## Project Structure
```
aus-market-analysis/
├── src/
│   ├── data/           # Data collection and processing
│   ├── models/         # ML models and training
│   ├── api/           # FastAPI endpoints
│   └── utils/         # Helper functions
├── tests/             # Unit and integration tests
├── notebooks/         # Jupyter notebooks for analysis
├── infra/            # Infrastructure as code
└── docs/             # Documentation
```

## Key Companies Analyzed
- Commonwealth Bank (CBA)
- BHP Group (BHP)
- CSL Limited (CSL)
- National Australia Bank (NAB)
- Westpac Banking Corp (WBC)
- ANZ Banking Group (ANZ)
- Woolworths Group (WOW)
- Wesfarmers (WES)
- Macquarie Group (MQG)
- Telstra (TLS)

## Technical Stack
- Python 3.9+
- FastAPI
- MLflow
- Pandas
- Scikit-learn
- PyTorch
- Docker
- Azure ML
- GitHub Actions

## Setup Instructions
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## API Endpoints
- `/api/v1/companies` - List all analyzed companies
- `/api/v1/predict/{symbol}` - Get predictions for a specific company
- `/api/v1/analysis/{symbol}` - Get detailed analysis
- `/api/v1/market/sentiment` - Get overall market sentiment

## Model Performance
- Prediction Accuracy: >70%
- API Response Time: <200ms
- Model Retraining: Daily

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 