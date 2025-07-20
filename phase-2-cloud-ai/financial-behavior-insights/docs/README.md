# Financial Behavior Insights - MLOps Pipeline

**Project**: Financial Behavior Insights  
**Date**: July 20, 2025  
**Status**: Production Ready  
**All Tests**: 4/4 PASSED  

## Overview

This project implements a production-ready MLOps pipeline for financial behavior analysis using Azure Machine Learning. The system processes banking data, trains machine learning models, and deploys them as managed online endpoints.

## Project Structure

```
financial-behavior-insights/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── training/          # Model training
│   ├── serving/           # Model deployment and scoring
│   └── utils/             # Utilities and configuration
├── data/                  # Data files
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── outputs/              # Model outputs and artifacts
├── tests/                # Test files
├── docs/                 # Documentation
├── workflows/            # Workflow automation
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Python dependencies
├── Makefile             # Automation commands
└── README.md            # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Azure CLI
- Azure ML workspace access
- Required Python packages (see requirements.txt)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd financial-behavior-insights
   ```

2. **Create virtual environment**
   ```bash
   python -m venv fin-envi
   source fin-envi/bin/activate  # On Windows: fin-envi\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Azure credentials**
   ```bash
   az login
   az account set --subscription <subscription-id>
   ```

### Usage

#### Core Operations

```bash
# Check deployment status
make status

# Run comprehensive tests
make test

# Quick health check
make health

# Start monitoring
make monitor
```

#### Full Pipeline

```bash
# Complete end-to-end pipeline
make full-pipeline

# Automated workflow execution
make workflow-runner
```

#### Advanced Operations

```bash
# Scale deployment
make scale

# Rollback deployment
make rollback

# Troubleshoot issues
make troubleshoot

# Clean up artifacts
make clean
```

## Architecture

### Components

1. **Data Processing** (`src/data/`)
   - Preprocesses banking data
   - Handles feature engineering
   - Manages data validation

2. **Model Training** (`src/training/`)
   - Trains machine learning models
   - Integrates with MLflow
   - Manages model versioning

3. **Model Serving** (`src/serving/`)
   - Deploys models to Azure ML
   - Handles scoring requests
   - Manages endpoint lifecycle

4. **Utilities** (`src/utils/`)
   - Configuration management
   - Monitoring and observability
   - Model registry operations

### Azure ML Resources

- **Workspace**: mlw-finance-phase-2
- **Endpoint**: fin-behavior-ep-fixed
- **Deployment**: blue
- **Model**: financial-behavior-model-fixed

## Testing

The project includes comprehensive testing at multiple levels:

1. **Environment Testing**: Verifies setup and dependencies
2. **Model Compatibility**: Tests model loading and prediction
3. **Local Testing**: Validates local model functionality
4. **Azure ML Testing**: Tests deployed endpoint integration

Run all tests:
```bash
make test
```

## Monitoring

The system includes comprehensive monitoring:

- Real-time health monitoring
- Performance metrics tracking
- Anomaly detection
- Automated alerting

## Configuration

Configuration is managed through environment variables and the `src/utils/config.py` module. Key configuration areas:

- Azure ML workspace settings
- Model training parameters
- Deployment configuration
- Monitoring settings

## Troubleshooting

Common issues and solutions are documented in the troubleshooting guides. For immediate help:

```bash
make troubleshoot
```

## Contributing

1. Follow the established code structure
2. Add tests for new functionality
3. Update documentation
4. Follow MLOps best practices

## License

This project is licensed under the MIT License.

## Support

For support and questions, please refer to the documentation or create an issue in the repository. 