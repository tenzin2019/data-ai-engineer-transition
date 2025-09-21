# Intelligent Document Analysis System

A comprehensive AI-powered document analysis platform deployed on Azure App Service, designed to extract insights from various document types including PDFs, Word documents, Excel files, and text files.

## 🚀 Live Application

**URL**: https://intelligent-document-analysis.azurewebsites.net

## ✨ Features

- **Multi-format Support**: PDF, DOCX, XLSX, TXT files up to 200MB
- **AI-Powered Analysis**: 
  - Text extraction and processing
  - Sentiment analysis
  - Key phrase extraction
  - Document summarization
  - Entity recognition
  - Confidence scoring
- **Persistent Storage**: SQLite database for analysis results
- **Real-time Processing**: Streamlit-based interactive interface
- **Analytics Dashboard**: Comprehensive insights and metrics
- **Data Management**: Clear analysis data functionality

## 🏗️ Architecture

### Technology Stack
- **Backend**: Python 3.11, Streamlit
- **AI/ML**: OpenAI GPT-4, NLTK, spaCy
- **Document Processing**: PyPDF2, python-docx, openpyxl
- **Database**: SQLite (with PostgreSQL support)
- **Deployment**: Azure App Service, Docker, Azure Container Registry
- **Infrastructure**: Azure CLI, Azure Resource Manager

### Project Structure
```
intelligent-document-analysis/
├── src/
│   ├── web/                 # Streamlit application
│   ├── core/                # Core processing modules
│   ├── services/            # Database and business logic
│   ├── utils/               # Utility functions
│   ├── models/              # Database models
│   └── api/                 # API endpoints
├── scripts/                 # Deployment and utility scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
├── Dockerfile.azure         # Azure-optimized Docker configuration
├── requirements-azure.txt   # Production dependencies
└── .azure/                  # Azure App Service configuration
```

## 🚀 Quick Start

### Prerequisites
- Azure CLI installed and configured
- Docker installed
- Python 3.11+ (for local development)

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd intelligent-document-analysis

# Install dependencies
pip install -r requirements-azure.txt

# Run the application
streamlit run src/web/app.py
```

### Azure Deployment
```bash
# Deploy to Azure
./scripts/deploy-azure.sh

# Verify deployment
./scripts/verify-deployment.sh
```

## 📋 Available Scripts

- `deploy-azure.sh` - Deploy application to Azure App Service
- `verify-deployment.sh` - Verify deployment status and health
- `housekeeping.sh` - Clean up unused files and optimize project
- `health_check.py` - Health check utility

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL` - Database connection string (default: SQLite)
- `DB_DISABLED` - Disable database operations (default: false)
- `OPENAI_API_KEY` - OpenAI API key for AI analysis
- `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` - Maximum file upload size (default: 200MB)

### Azure App Service Settings
- Port: 8000
- Platform: Linux/AMD64
- Container: Docker
- Storage: Enabled

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test categories
pytest tests/test_document_processor.py
pytest tests/test_azure_integration.py
```

## 📊 Monitoring

- **Health Endpoint**: `/health`
- **Application Logs**: Available through Azure Portal
- **Metrics**: Azure App Service metrics and insights

## 🔒 Security

- Non-root user execution in Docker container
- Input validation and sanitization
- File type validation
- Secure environment variable handling
- CORS and XSRF protection configured

## 🚀 Performance

- Optimized Docker layers for faster builds
- Efficient document processing pipeline
- Caching for NLTK data
- Streamlit optimizations for Azure App Service

## 📈 Scalability

- Horizontal scaling through Azure App Service
- Stateless application design
- Database-agnostic architecture
- Container-based deployment

## 🛠️ Troubleshooting

### Common Issues
1. **File Upload Errors**: Check file size limits and format support
2. **Analysis Failures**: Verify OpenAI API key configuration
3. **Database Issues**: Check SQLite file permissions
4. **Deployment Issues**: Verify Azure credentials and resource availability

### Logs
- Application logs available through Azure Portal
- Container logs accessible via Azure CLI
- Health check endpoint for status monitoring

## 📝 License

This project is part of the Data AI Engineer Transition portfolio.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Azure App Service logs
- Contact the development team

---

**Status**: ✅ Production Ready  
**Last Updated**: December 2024  
**Version**: 1.0.0