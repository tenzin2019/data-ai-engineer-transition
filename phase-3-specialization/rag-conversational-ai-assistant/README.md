# RAG Conversational AI Assistant

A comprehensive Retrieval-Augmented Generation (RAG) system with conversational AI capabilities, designed for enterprise-grade question-answering with advanced features including LLM orchestration, prompt versioning, human-in-the-loop feedback, model observation, and drift detection.

## Project Overview

The RAG Conversational AI Assistant is an enterprise-grade question-answering platform that combines advanced retrieval techniques with large language models to provide accurate, contextual, and traceable answers. This system incorporates industry best practices and cutting-edge AI technologies for continuous improvement and optimal performance.

## Key Features

### Core RAG Capabilities
- **Intelligent Document Retrieval**: Multi-format document processing (PDF, DOCX, TXT, HTML)
- **Semantic Search**: Advanced vector-based document search with hybrid retrieval
- **Context-Aware Generation**: LLM-powered answer generation with source attribution
- **Multi-Modal Support**: Text, image, and audio content processing

### Advanced AI Features
- **LLM Orchestration**: Dynamic model selection and load balancing
- **Prompt Versioning**: A/B testing and optimization of prompt templates
- **Human-in-the-Loop**: Expert review and feedback integration
- **Model Monitoring**: Real-time performance tracking and drift detection
- **Conversational Memory**: Context-aware multi-turn conversations

### Enterprise Features
- **Scalable Architecture**: Microservices-based design for high availability
- **Security & Compliance**: Enterprise-grade security with audit trails
- **API Integration**: RESTful APIs for seamless integration
- **Analytics Dashboard**: Comprehensive insights and performance metrics
- **Multi-Tenant Support**: Isolated environments for different organizations

## Architecture

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   API Gateway   │    │   Load Balancer │
│   (React/Vue)   │◄──►│   (FastAPI)     │◄──►│   (Nginx)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Orchestration Layer                     │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Query Router   │  Prompt Manager │  Model Selector │  Cache  │
│  & Preprocessor │  & Versioning   │  & Load Balancer│ Manager │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI/ML Processing Layer                      │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Embedding      │  Vector Store   │  LLM Orchestrator│  Model  │
│  Engine         │  (Pinecone/     │  (LangChain/    │ Monitor │
│  (OpenAI/       │  Weaviate)      │  LlamaIndex)    │ & Track │
│  Sentence-BERT) │                 │                 │         │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                        │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Document Store │  Vector DB      │  Feedback DB    │  Audit  │
│  (PostgreSQL/   │  (Pinecone/     │  (MongoDB)      │  Logs   │
│  Elasticsearch) │  Weaviate)      │                 │ (S3)    │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
```

## Technology Stack

### Backend Technologies
- **API Framework**: FastAPI with async support
- **LLM Orchestration**: LangChain (primary framework)
- **Vector Database**: Pinecone, Weaviate, Chroma, or Azure AI Search
- **Document Store**: PostgreSQL with full-text search
- **Cache**: Redis for response caching
- **Message Queue**: Apache Kafka for event streaming

### AI/ML Technologies
- **LLM Orchestration**: LangChain with multi-provider support
- **Embedding Models**: OpenAI text-embedding-ada-002, Sentence-BERT, Azure OpenAI
- **LLM Providers**: OpenAI GPT-4, Anthropic Claude, Azure OpenAI
- **Vector Search**: FAISS, Pinecone, Weaviate, Azure AI Search
- **ML Monitoring**: Weights & Biases, MLflow, Evidently AI
- **Drift Detection**: Evidently AI, Alibi Detect, custom algorithms

### Frontend Technologies
- **Framework**: React.js with TypeScript
- **UI Components**: Material-UI or Ant Design
- **State Management**: Redux Toolkit or Zustand
- **Real-time**: WebSocket for live updates
- **Visualization**: D3.js, Plotly.js for analytics

### Infrastructure & DevOps
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Kubernetes with Helm charts, Azure App Service
- **CI/CD**: GitHub Actions with automated testing and Azure deployment
- **Monitoring**: Prometheus, Grafana, Jaeger, Azure Application Insights
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana), Azure Log Analytics
- **Storage**: Azure Blob Storage, Azure Files
- **Cloud Platform**: Microsoft Azure with native service integration

## Project Structure

```
rag-conversational-ai-assistant/
├── docs/                           # Comprehensive documentation
│   ├── RAG_QA_PROJECT_PLAN.md     # Main project plan
│   ├── LLM_ORCHESTRATION_ARCHITECTURE.md
│   ├── PROMPT_VERSIONING_SYSTEM.md
│   ├── HUMAN_IN_THE_LOOP_SYSTEM.md
│   ├── MODEL_OBSERVATION_FRAMEWORK.md
│   └── DRIFT_DETECTION_SYSTEM.md
├── src/                           # Source code
│   ├── api/                       # FastAPI backend
│   ├── core/                      # Core RAG functionality
│   ├── orchestration/             # LLM orchestration
│   ├── monitoring/                # Model monitoring
│   ├── feedback/                  # Human-in-the-loop
│   ├── frontend/                  # React frontend
│   └── utils/                     # Utility functions
├── tests/                         # Test suite
├── deployments/                   # Kubernetes manifests
├── scripts/                       # Deployment scripts
├── config/                        # Configuration files
├── requirements.txt               # Python dependencies
├── package.json                   # Node.js dependencies
├── docker-compose.yml             # Local development
└── README.md                      # This file
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (recommended)
- API keys for LLM services (OpenAI, Anthropic, or Azure OpenAI)

### Method 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd rag-conversational-ai-assistant

# Run automated setup
chmod +x scripts/setup.sh
./scripts/setup.sh

# Test the setup
python test_setup.py

# Edit environment variables
cp env.example .env
nano .env  # Add your API keys

# Start with Docker Compose (easiest)
docker-compose up -d

# Access the application
# Frontend: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Method 2: Manual Setup
```bash
# Clone the repository
git clone <repository-url>
cd rag-conversational-ai-assistant

# Create Python environment
python3 -m venv rag-venv
source rag-venv/bin/activate  # On Windows: rag-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys and configuration

# Create necessary directories
mkdir -p uploads chroma_db logs temp

# Test the setup
python test_setup.py

# Start the backend API
python src/api/main.py

# In another terminal, start the frontend
source rag-venv/bin/activate
streamlit run src/frontend/streamlit_app.py
```

### Method 3: Development with Individual Services
```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Set environment variables for local development
export DATABASE_URL="postgresql://postgres:password@localhost:5432/rag_assistant"
export REDIS_URL="redis://localhost:6379/0"

# Run the API
python src/api/main.py

# Run the frontend
streamlit run src/frontend/streamlit_app.py
```

### Production Deployment

#### Azure App Service Deployment
```bash
# Deploy to Azure App Service
./scripts/deploy-azure.sh

# Or deploy manually
az webapp deployment source config \
  --name rag-assistant-api \
  --resource-group rag-assistant-rg \
  --repo-url https://github.com/your-org/rag-conversational-ai-assistant \
  --branch main
```

#### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f deployments/

# Or use Helm
helm install rag-assistant ./helm-chart
```

## Performance Targets

- **Response Time**: <2 seconds for 95% of queries
- **Throughput**: 1000+ queries per minute
- **Availability**: 99.9% uptime
- **Accuracy**: >95% retrieval accuracy, >90% answer quality
- **Scalability**: Support 1000+ concurrent users

## Security & Compliance

- **Data Encryption**: End-to-end encryption for data in transit and at rest
- **Access Control**: Role-based access control (RBAC) with fine-grained permissions
- **Data Privacy**: GDPR and CCPA compliance with data anonymization
- **Audit Logging**: Comprehensive audit trails for all operations
- **Model Security**: Input validation, output filtering, and prompt injection protection

## Monitoring & Analytics

- **Real-time Monitoring**: Live performance dashboards
- **Automated Alerting**: Proactive alerting for issues
- **Performance Tracking**: Historical performance analysis
- **User Analytics**: User behavior and satisfaction analysis
- **Model Drift Detection**: Automated detection of model performance degradation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of the Data AI Engineer Transition portfolio.

## Support

For issues and questions:
- Check the documentation in the `docs/` folder
- Review the troubleshooting guides
- Contact the development team
- Create an issue in the repository

---

**Status**: In Development  
**Last Updated**: December 2024  
**Version**: 0.1.0  
**Maintainer**: Data AI Engineer Transition Team