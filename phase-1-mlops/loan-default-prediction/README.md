# Loan Default Prediction – End-to-End MLOps Pipeline

## Overview

This project demonstrates a production-grade MLOps workflow for a loan default prediction use case, built from local experimentation to cloud deployment using **Azure ML** and industry-standard best practices.\
It features containerized training, automated CI/CD, cloud model deployment, endpoint testing, and clear documentation for future scaling and handover.

---

## Project Structure

```
loan-default-prediction/
├── src/
│   ├── training/
│   │   ├── train_model.py
│   │   └── deploy_model.py
│   └── api/              # (optional: future API serving)
├── tests/
│   └── test_endpoint.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone and Setup

```bash
git clone https://github.com/yourhandle/senior-ai-transition.git
cd senior-ai-transition/phase-1-mlops/loan-default-prediction
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your Azure credentials and workspace info:

```
AZURE_CLIENT_ID=your-azure-client-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_SECRET=your-secret
AZURE_SUBSCRIPTION_ID=your-sub-id
AZURE_RESOURCE_GROUP=your-rg
AZURE_WORKSPACE_NAME=your-ws
```

### 3. Build and Test Locally with Docker

```bash
docker build -t loan-predictor-app:latest .
docker run --rm --env-file .env loan-predictor-app:latest python src/training/train_model.py
```

---

## CI/CD Workflow

- Automated pipeline (GitHub Actions) on pushes to `main`, `dev`, or `feature/**`.
- Steps:
  1. Checkout and test code
  2. Build Docker image and train model in container
  3. Deploy to Azure ML endpoint (on main/dev)
  4. Integration test: endpoint is called and response validated
- Pipeline config: `.github/workflows/ci-cd.yml`

---

## Monitoring

- Endpoint health/status: View in **Azure ML Studio → Endpoints**
- Logs & deployment history: Click endpoint, view activity and logs tabs
- For advanced telemetry, see [Azure ML Monitoring](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-endpoints-app-insights?view=azureml-api-2)

---

## Secrets Management

- **Never commit secrets.** Use environment variables.
- Use GitHub Secrets for CI/CD, and a local `.env` for development/testing.

---

## Troubleshooting

| Issue                      | Solution                                              |
| -------------------------- | ----------------------------------------------------- |
| Docker build slow          | Use `.dockerignore`, pin versions in requirements.txt |
| Missing env variable error | Check that all `AZURE_*` vars are set in `.env`       |
| Auth failure on Azure ML   | Double-check Service Principal and workspace config   |
| Model/endpoint not found   | Confirm deployment finished successfully in Azure ML  |

---

## Extending

- Add FastAPI or Flask to `src/api/` for REST serving.
- Integrate SHAP/LIME for explainability.
- Use DVC for data versioning.
- Add infra-as-code with Terraform/Bicep for cloud provisioning.
- Enhance with Application Insights for telemetry and alerting.

---

## Contact

For questions, reach out on [LinkedIn](https://linkedin.com/in/tenzin-jamyang)\
or open an issue in this repo.

---

