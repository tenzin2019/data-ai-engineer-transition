# Financial Behavior Insights - Azure ML Deployment

A comprehensive MLOps project for predicting high-amount transactions in banking data using Azure Machine Learning.

## 🎯 Project Overview

This project demonstrates a complete end-to-end ML pipeline including:
- Data preprocessing and feature engineering
- Model training with scikit-learn
- Model compatibility fixes for Azure ML deployment
- Azure ML managed online endpoint deployment
- Comprehensive testing and validation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Preprocessing  │───▶│  Model Training │
│   (CSV)         │    │   (Python)      │    │  (sklearn)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Azure ML      │◀───│  Deployment     │◀───│  Model Retrain  │
│   Endpoint      │    │   Manager       │    │  (Compatible)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Testing &     │
│   Validation    │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Azure ML Workspace** with the following resources:
   - Subscription ID
   - Resource Group
   - Workspace Name

2. **Environment Variables** (create a `.env` file):
   ```bash
   AZURE_SUBSCRIPTION_ID=your_subscription_id
   AZURE_RESOURCE_GROUP=your_resource_group
   AZURE_WORKSPACE_NAME=your_workspace_name
   ```

3. **Python Environment**:
   ```bash
   # Create conda environment
   make create-env
   
   # Activate environment
   source fin-envi/bin/activate
   
   # Install dependencies
   make install
   ```

### Complete End-to-End Pipeline

Run the entire pipeline with one command:

```bash
# Option 1: Using Makefile
make full-pipeline

# Option 2: Using Python workflow runner (recommended)
make workflow-runner

# Option 3: Direct Python execution
python3 workflow_runner.py --full-pipeline
```

This will execute:
1. ✅ Data preparation and preprocessing
2. ✅ Model training with original sklearn version
3. ✅ Model retraining for Azure ML compatibility
4. ✅ Deployment to Azure ML managed endpoint
5. ✅ Comprehensive testing and validation

## 📁 Project Structure

```
financial-behavior-insights/
├── data/
│   ├── Banking-Dataset/           # Raw data
│   └── processed/                 # Preprocessed data
├── src/
│   ├── data/
│   │   └── preprocess_banking.py  # Data preprocessing
│   ├── training/
│   │   └── train_model.py         # Model training
│   ├── serving/
│   │   ├── deploy_manager.py      # Deployment manager
│   │   └── score.py              # Scoring script
│   └── utils/
│       └── model_registry.py      # Model registration
├── outputs/                       # Model artifacts
├── tests/                         # Test files
├── workflow_runner.py             # Main workflow orchestrator
├── retrain_compatible_model.py    # Model compatibility fix
├── test_deployments.py            # Deployment testing
├── Makefile                       # Build automation
├── requirements.txt               # Python dependencies
└── environment.yml                # Conda environment
```

## 🔧 Available Commands

### Makefile Commands

```bash
# Environment setup
make create-env          # Create conda environment
make update-env          # Update conda environment
make install             # Install dependencies
make check-env           # Check environment setup

# Data and model
make data-prep           # Prepare and preprocess data
make train              # Train model with original sklearn
make retrain            # Retrain model for Azure ML compatibility

# Deployment
make deploy             # Deploy model to Azure ML
make status             # Check deployment status
make logs               # Get deployment logs
make test               # Test deployments

# Workflows
make full-pipeline      # Complete end-to-end workflow
make workflow-runner    # Run with Python workflow runner
make quick-deploy       # Quick deployment (assumes model ready)

# Utilities
make clean              # Clean artifacts and logs
make troubleshoot       # Run troubleshooting checks
make reset              # Reset everything and start fresh
```

### Python Scripts

```bash
# Workflow runner
python3 workflow_runner.py --full-pipeline
python3 workflow_runner.py --step data
python3 workflow_runner.py --step train
python3 workflow_runner.py --step retrain
python3 workflow_runner.py --step deploy
python3 workflow_runner.py --step test

# Individual scripts
python3 src/data/preprocess_banking.py --input data/Banking-Dataset/Comprehensive_Banking_Database.csv --output data/processed/
python3 src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv --output-dir outputs/
python3 retrain_compatible_model.py
python3 src/serving/deploy_manager.py --action deploy --model-name financial-behavior-model-fixed
python3 test_deployments.py --test-type all
```

## 🔍 Key Features

### 1. **Model Compatibility Fix**
- Automatically detects scikit-learn version incompatibilities
- Retrains model with Azure ML-compatible versions (1.0.2 or 1.1.3)
- Handles tree structure differences between sklearn versions

### 2. **Robust Deployment**
- Uses Azure ML managed online endpoints
- Implements blue-green deployment strategy
- Handles model registration and versioning
- Custom scoring script for inference

### 3. **Comprehensive Testing**
- Environment validation
- Model compatibility testing
- Local model testing
- Azure ML endpoint testing
- Deployment status monitoring

### 4. **Error Handling**
- Detailed logging throughout the pipeline
- Graceful failure handling
- Automatic cleanup and recovery
- Timeout protection for long-running operations

## 🐛 Troubleshooting

### Common Issues

1. **"User container has crashed or terminated"**
   - **Cause**: Scikit-learn version incompatibility
   - **Solution**: Run `make retrain` to create compatible model

2. **Missing environment variables**
   - **Cause**: Azure ML configuration not set
   - **Solution**: Create `.env` file with required variables

3. **Model loading errors**
   - **Cause**: Model path or format issues
   - **Solution**: Check model files in `outputs/` directory

4. **Deployment failures**
   - **Cause**: Azure ML service issues or configuration problems
   - **Solution**: Check logs with `make logs` and verify Azure ML setup

### Debugging Commands

```bash
# Check environment
make check-env

# Troubleshoot issues
make troubleshoot

# Get detailed logs
make logs

# Test specific components
python3 test_deployments.py --test-type environment
python3 test_deployments.py --test-type compatibility
python3 test_deployments.py --test-type azure
```

## 📊 Model Information

### Model Details
- **Algorithm**: Random Forest Classifier
- **Target**: High-amount transaction prediction (binary classification)
- **Features**: 12 engineered features including:
  - Demographics (Age, Gender)
  - Transaction details (Amount, Type, Hour, Day)
  - Account information (Balance, Age)
- **Performance**: ~85% accuracy on test set

### Model Artifacts
- `outputs/model_compatible.joblib`: Complete model with scaler and metadata
- `outputs/model_info.json`: Model information and feature list
- `simple_model/model.pkl`: Simple model for basic deployment

## 🔄 Deployment Lifecycle

1. **Development**: Local training and testing
2. **Compatibility**: Model retraining for Azure ML
3. **Registration**: Model registration in Azure ML workspace
4. **Deployment**: Endpoint creation and model deployment
5. **Validation**: Comprehensive testing and monitoring
6. **Production**: Endpoint ready for inference

## 📈 Monitoring and Logs

### Log Files
- `workflow.log`: Complete workflow execution log
- Azure ML deployment logs: Available via `make logs`

### Monitoring Points
- Model training metrics and performance
- Deployment status and health
- Endpoint response times and errors
- Data drift and model degradation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `workflow.log`
3. Run `make troubleshoot` for automated diagnostics
4. Check Azure ML service status

---

**Last Updated**: July 19, 2025  
**Status**: ✅ Production Ready  
**Azure ML Endpoint**: `fin-behavior-ep-fixed` (Succeeded)
