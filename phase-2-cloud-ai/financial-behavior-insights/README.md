# Financial Behavior Insights - ML Training Pipeline

A robust machine learning pipeline for predicting high-amount financial transactions using Random Forest classification.

## 🚀 Features

- **Comprehensive Data Validation**: Validates data quality, types, and structure
- **Memory-Efficient Processing**: Supports chunked loading for large datasets
- **Robust Error Handling**: Graceful failure handling with detailed logging
- **Azure ML Integration**: Model registration and deployment ready
- **MLflow Tracking**: Experiment tracking and model versioning
- **Configurable Parameters**: Flexible hyperparameter tuning and training options
- **Production-Ready**: Comprehensive testing and validation

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd financial-behavior-insights
```

2. **Create virtual environment**:
```bash
# Using conda
conda env create -f environment.yml
conda activate financial-behavior-inference

# Or using pip
python -m venv fin-env
source fin-env/bin/activate  # On Windows: fin-env\Scripts\activate
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the project root with:
```bash
# Azure ML Configuration (required for model registration)
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name

# Optional configurations
RANDOM_STATE=42
N_ITER=20
CV_FOLDS=3
TEST_SIZE=0.2
```

## 📊 Usage

### Basic Training

```bash
python src/training/train_model.py \
    --input-data data/processed/Comprehensive_Banking_Database_processed.csv \
    --output-dir outputs \
    --random-state 42
```

### Advanced Training with Custom Parameters

```bash
python src/training/train_model.py \
    --input-data data/processed/Comprehensive_Banking_Database_processed.csv \
    --output-dir outputs \
    --random-state 42 \
    --n-iter 50 \
    --cv 5 \
    --test-size 0.25 \
    --chunk-size 10000 \
    --register-model \
    --model-name "my-custom-model" \
    --model-description "Custom Random Forest for financial insights"
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-data` | str | **Required** | Path to input CSV file |
| `--output-dir` | str | `outputs` | Directory to save model and metrics |
| `--random-state` | int | `42` | Random state for reproducibility |
| `--n-iter` | int | `20` | Number of hyperparameter tuning iterations |
| `--cv` | int | `3` | Cross-validation folds |
| `--test-size` | float | `0.2` | Proportion of data for testing |
| `--chunk-size` | int | `None` | Chunk size for large CSV files |
| `--register-model` | flag | `False` | Register model in Azure ML |
| `--model-name` | str | `financial-behavior-insights-model` | Azure ML model name |
| `--model-description` | str | `Random Forest for HighAmount prediction` | Model description |

## 🔧 Data Requirements

### Input Data Format

The pipeline expects a CSV file with the following requirements:

- **Target Column**: Must contain a column named `HighAmount` (binary: 0 or 1)
- **Feature Columns**: All other columns will be used as features
- **Data Types**: Target column must be numeric
- **Missing Values**: No missing values in target column
- **Duplicate Columns**: No duplicate column names

### Example Data Structure

```csv
feature1,feature2,feature3,HighAmount
1.2,0.5,0.8,0
2.1,1.2,0.3,1
0.8,0.9,1.1,0
...
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_train_model.py::TestDataValidation -v
pytest tests/test_train_model.py::TestEnvironmentValidation -v
pytest tests/test_train_model.py::TestIntegration -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Output Files

After successful training, the following files are generated:

```
outputs/
├── model.joblib          # Trained model (serialized)
├── metrics.json          # Training metrics
└── mlruns/              # MLflow experiment tracking
```

### Metrics Format

```json
{
  "accuracy": 0.85,
  "precision": 0.82,
  "recall": 0.78,
  "f1_score": 0.80,
  "roc_auc": 0.88
}
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Environment Variables Missing
```
ValueError: Missing required environment variables: ['AZURE_SUBSCRIPTION_ID', 'AZURE_RESOURCE_GROUP', 'AZURE_WORKSPACE_NAME']
```
**Solution**: Set up your `.env` file with the required Azure ML credentials.

#### 2. Data Validation Errors
```
ValueError: Target column 'HighAmount' not found in dataset
```
**Solution**: Ensure your CSV file contains a column named `HighAmount`.

#### 3. Memory Issues with Large Files
```
MemoryError: Unable to allocate array
```
**Solution**: Use the `--chunk-size` parameter for large datasets:
```bash
python src/training/train_model.py --input-data large_file.csv --chunk-size 10000
```

#### 4. MLflow Connection Issues
```
Warning: MLflow setup failed. Continuing without MLflow tracking.
```
**Solution**: This is not critical - the pipeline will continue without MLflow tracking.

#### 5. Azure ML Registration Failures
```
Azure ML model registration failed: [error details]
```
**Solution**: 
- Verify Azure credentials and permissions
- Check if the model file exists
- Ensure Azure ML workspace is accessible

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

- **Environment Variables**: Never commit `.env` files to version control
- **Azure Credentials**: Use managed identities or service principals in production
- **Data Privacy**: Ensure compliance with data protection regulations
- **Model Security**: Validate model inputs and outputs in production

## 📝 Recent Fixes

### Version 2.0 - Major Improvements

#### Critical Bug Fixes
- ✅ **Environment Variable Validation**: Added comprehensive validation for Azure ML credentials
- ✅ **Silent Failure Prevention**: Proper error propagation and handling
- ✅ **Data Validation**: Comprehensive data quality checks
- ✅ **Memory Management**: Chunked loading for large datasets
- ✅ **Model Validation**: Pre-save validation to ensure model integrity

#### Enhanced Features
- ✅ **Configurable Parameters**: All training parameters are now configurable
- ✅ **Robust Error Handling**: Graceful failure handling throughout the pipeline
- ✅ **Comprehensive Logging**: Detailed logging for debugging and monitoring
- ✅ **Production-Ready**: Extensive testing and validation
- ✅ **Azure ML Integration**: Improved model registration with proper error handling

#### New Capabilities
- ✅ **Large File Support**: Memory-efficient processing for datasets >100MB
- ✅ **Flexible Output**: Configurable output directory and file formats
- ✅ **Model Registration**: Optional Azure ML model registry integration
- ✅ **MLflow Integration**: Robust experiment tracking setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Open an issue with detailed error information
4. Include your environment details and data format
