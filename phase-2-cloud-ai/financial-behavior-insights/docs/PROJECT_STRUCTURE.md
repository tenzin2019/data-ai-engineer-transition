# Project Structure - Financial Behavior Insights

## Overview

This document describes the clean, organized structure of the Financial Behavior Insights MLOps project after restructuring and cleanup.

## Directory Structure

```
financial-behavior-insights/
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   └── preprocess_banking.py # Banking data preprocessing
│   ├── training/                 # Model training modules
│   │   ├── train_model.py        # Main training script
│   │   ├── mlflow_model.py       # MLflow integration
│   │   └── utils.py              # Training utilities
│   ├── serving/                  # Model serving modules
│   │   ├── deploy_manager.py     # Deployment management
│   │   ├── score.py              # Scoring script
│   │   └── endpoint.yml          # Endpoint configuration
│   └── utils/                    # Utility modules
│       ├── config.py             # Configuration management
│       ├── monitoring.py         # Monitoring and observability
│       ├── fix_model_format.py   # Model format utilities
│       └── model_registry.py     # Model registry operations
├── data/                         # Data files
│   ├── Banking-Dataset/          # Raw banking data
│   └── processed/                # Processed data files
├── outputs/                      # Model outputs and artifacts
│   ├── model_compatible.joblib   # Azure ML compatible model
│   └── model_info.json          # Model metadata
├── tests/                        # Test files
│   ├── test_connection.py        # Connection tests
│   ├── test_core_functions.py    # Core function tests
│   ├── test_endpoint.py          # Endpoint tests
│   ├── test_integration.py       # Integration tests
│   └── test_train_model.py       # Training tests
├── docs/                         # Documentation
│   ├── README.md                 # Documentation README
│   ├── PROJECT_SUMMARY.md        # Project summary
│   └── PROJECT_STRUCTURE.md      # This file
├── workflows/                    # Workflow automation
│   ├── blue_green_deploy.py      # Blue-green deployment
│   ├── config.yaml              # Workflow configuration
│   ├── ml_workflow.py           # ML workflow orchestration
│   └── README.md                # Workflow documentation
├── notebooks/                    # Jupyter notebooks
├── monitoring/                   # Monitoring data
├── workflow_artifacts/           # Workflow execution artifacts
├── fin-envi/                     # Virtual environment
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── Makefile                      # Automation commands
├── test_deployments.py           # Comprehensive testing
├── workflow_runner.py            # Workflow orchestration
├── retrain_compatible_model.py   # Model compatibility fix
├── workflow.log                  # Workflow execution log
├── mlflow.db                     # MLflow database
├── config.json                   # Configuration file
├── .gitignore                    # Git ignore rules
├── .dvcignore                    # DVC ignore rules
└── README.md                     # Main project README
```

## Key Components

### Source Code (`src/`)

#### Data Processing (`src/data/`)
- **preprocess_banking.py**: Handles banking data preprocessing with scikit-learn 1.1.3 compatibility

#### Model Training (`src/training/`)
- **train_model.py**: Main model training script with MLflow integration
- **mlflow_model.py**: MLflow model logging and registration
- **utils.py**: Training utilities and helpers

#### Model Serving (`src/serving/`)
- **deploy_manager.py**: Azure ML deployment management
- **score.py**: Scoring script for Azure ML endpoints
- **endpoint.yml**: Endpoint configuration

#### Utilities (`src/utils/`)
- **config.py**: Centralized configuration management
- **monitoring.py**: Monitoring and observability features
- **fix_model_format.py**: Model format utilities
- **model_registry.py**: Model registry operations

### Data (`data/`)
- **Banking-Dataset/**: Raw banking data files
- **processed/**: Preprocessed data files ready for training

### Outputs (`outputs/`)
- **model_compatible.joblib**: Azure ML compatible model
- **model_info.json**: Model metadata and information

### Tests (`tests/`)
Comprehensive test suite covering:
- Connection testing
- Core function testing
- Endpoint testing
- Integration testing
- Training testing

### Documentation (`docs/`)
- **README.md**: Documentation overview
- **PROJECT_SUMMARY.md**: Complete project summary
- **PROJECT_STRUCTURE.md**: This structure document

### Workflows (`workflows/`)
- **blue_green_deploy.py**: Blue-green deployment strategy
- **config.yaml**: Workflow configuration
- **ml_workflow.py**: ML workflow orchestration
- **README.md**: Workflow documentation

### Configuration Files

#### Root Level
- **requirements.txt**: Python package dependencies
- **environment.yml**: Conda environment specification
- **Makefile**: Automation commands and workflows
- **config.json**: Project configuration
- **.gitignore**: Git ignore patterns
- **.dvcignore**: DVC ignore patterns

#### Main Scripts
- **test_deployments.py**: Comprehensive deployment testing
- **workflow_runner.py**: Workflow orchestration
- **retrain_compatible_model.py**: Model compatibility fixes

## File Naming Conventions

### Python Files
- Use snake_case for file names
- Descriptive names that indicate functionality
- Separate words with underscores

### Configuration Files
- Use descriptive names with appropriate extensions
- YAML for structured configuration
- JSON for simple key-value pairs

### Documentation Files
- Use UPPER_CASE for main documentation files
- Descriptive names that indicate content
- Markdown format for all documentation

## Cleanup Actions Performed

### Removed Files
- Deprecated documentation files with emojis
- Redundant summary files
- Temporary verification scripts
- Old checkpoint files

### Restructured
- Created organized documentation directory
- Cleaned up Makefile (removed emojis)
- Standardized file naming
- Improved project structure

### Maintained
- All functional source code
- Essential configuration files
- Core automation scripts
- Working test suite

## Best Practices Implemented

### Code Organization
- Clear separation of concerns
- Modular design
- Consistent naming conventions
- Proper import structure

### Documentation
- Clean, professional documentation
- No emojis in code or documentation
- Comprehensive project structure
- Clear usage instructions

### Configuration
- Centralized configuration management
- Environment-specific settings
- Secure credential handling
- Validation and error handling

### Testing
- Comprehensive test coverage
- Multi-level testing strategy
- Automated test execution
- Clear test organization

## Usage

### Development
```bash
# Setup environment
make setup

# Run tests
make test

# Full pipeline
make full-pipeline
```

### Production
```bash
# Deploy to production
make deploy

# Monitor deployment
make monitor

# Health check
make health
```

### Maintenance
```bash
# Clean up artifacts
make clean

# Troubleshoot issues
make troubleshoot

# Update dependencies
make update-deps
```

## Conclusion

The project structure has been cleaned and organized following MLOps best practices. All deprecated files have been removed, emojis have been eliminated from code and documentation, and the structure is now production-ready and maintainable.

---

*Last Updated: July 20, 2025* 