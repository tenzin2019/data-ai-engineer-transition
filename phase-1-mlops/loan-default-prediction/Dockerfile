# 🐳 Dockerfile for FastAPI ML App (Loan Default Prediction)
# Optimized for local dev and Azure App Service deployment

# Step 1: Use official Python image
FROM python:3.11-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Install OS dependencies
# Install system dependencies (if you need e.g., build-essential, git, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for Docker cache efficiency)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your project files into the container
COPY ./src ./src

# Optional: Expose port if you're running an API server
EXPOSE 8000

# Default command: run tests (change this as needed)
CMD ["pytest", "tests/"]