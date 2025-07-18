# MLOps Best Practices Checklist

This checklist summarizes industry best practices for deploying, monitoring, and maintaining machine learning models in production, with a focus on AzureML and MLflow workflows.

---

## 1. Model Deployment
- [ ] **Use containerization** (e.g., Docker) for portability and reproducibility.
- [ ] **Automate deployments** with CI/CD pipelines (e.g., GitHub Actions, Azure Pipelines).
- [ ] **Blue-Green/Canary/Shadow Deployment**
    - Blue-Green: Deploy new model alongside current, switch traffic when ready.
    - Canary: Gradually shift a percentage of traffic to the new model.
    - Shadow: Run new model in parallel, log predictions for validation.
- [ ] **No-code MLflow deployment**: Use AzureML’s no-code deployment for MLflow models to avoid custom scoring scripts and environments.

## 2. Model Registry and Versioning
- [ ] Register all models in a central registry (AzureML Model Registry).
- [ ] Track model versions, metadata, training data, and code.
- [ ] Automate promotion of models from staging to production based on test results.

## 3. Monitoring and Logging
- [ ] Monitor model performance (accuracy, latency, drift) in production.
- [ ] Log all predictions, inputs, and outputs for traceability and debugging.
- [ ] Set up alerts for anomalies or performance degradation.

## 4. Governance and Compliance
- [ ] Maintain full lineage: Track data, code, and model versions for every deployment.
- [ ] Document model decisions and ensure explainability.
- [ ] Automate audits and ensure all artifacts are versioned and accessible.

## 5. Scalability and Reliability
- [ ] Design for scale: Use managed endpoints and autoscaling features.
- [ ] Test under load (e.g., with Locust) to ensure low latency.
- [ ] Automate rollbacks for quick recovery.

## 6. Security
- [ ] Store secrets securely (Azure Key Vault, environment variables).
- [ ] Use RBAC to restrict access to endpoints and resources.
- [ ] Use network isolation (VNETs, private endpoints) for sensitive workloads.

## 7. Documentation and Collaboration
- [ ] Keep README, architecture diagrams, and usage instructions up to date.
- [ ] Collaborate via code reviews and shared notebooks.

## 8. Continuous Improvement
- [ ] Automate retraining pipelines as new data arrives.
- [ ] Monitor for data drift and trigger retraining or alerts as needed.

---

## References
- [KDnuggets: MLOps Best Practices](https://www.kdnuggets.com/2021/07/mlops-best-practices.html)
- [AzureML: Safe rollout/blue-green deployment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-safely-rollout-online-endpoints?view=azureml-api-2&tabs=azure-cli)
- [Enable Shadow Deployment](https://se-ml.github.io/best_practices/04-shadow_models_prod/)
- [Advanced ML Model Deployment Techniques](https://the-ml-engineer-guy.medium.com/mastering-advanced-ml-model-deployment-techniques-2e1618b60f0c) 