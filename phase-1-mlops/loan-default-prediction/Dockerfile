# 🐳 Dockerfile for FastAPI ML App (Loan Default Prediction)
# Optimized for local dev and Azure App Service deployment


# Step 1: Use official Python image
FROM python:3.11-slim

# Set environment variable to indicate Docker environment
ENV RUNNING_IN_DOCKER=1

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirement files and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy app source code into container
COPY ./src ./src/

# Step 6: Copy MLflow model runs into container
COPY ./mlruns ./mlruns/

# Step 7: Expose port
EXPOSE 8000

# Step 8: Set the command to run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]