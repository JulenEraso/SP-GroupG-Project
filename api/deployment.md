# Breast Cancer Diagnosis API - Deployment Guide

This document outlines the steps to containerise, deploy, and test the Breast Cancer Prediction API using either **Docker** or **Podman**.

## Project Structure
- `main.py`: FastAPI application logic
- `model_pipeline.pkl`: Trained pipeline (Imputer, MinMaxScaler, Logistic Regression)
- `Dockerfile`: Container build instructions
- `requirements.txt`: Python package dependencies

## Deployment Steps

### 1. Environment Setup
- **For Podman:** Run `podman machine start` in PowerShell
- **For Docker:** Ensure Docker Desktop is running

### 2. Building the Image
Navigate to the `/api` directory and run the command:

**Podman:** `podman build -t breast-cancer-api .`
**Docker:** `docker build -t breast-cancer-api .`

### 3. Running the Container
Map host port **8080** to container port **8000** to avoid Windows system conflicts

**Podman:**
`podman run -d -p 8080:8000 --name diagnosis-api breast-cancer-api`

**Docker:**
`docker run -d -p 8080:8000 --name diagnosis-api breast-cancer-api`

## Testing and Validation

### Accessing the API
Once the container is running, the API can be accessed via any web browser on the host machine:
- **Home Page:** `http://localhost:8080/`
- **Interactive API:** `http://localhost:8080/docs`

Values can be entered by expanding the `/predict` section and clicking "Try it out".

### Automated Test (PowerShell)
To test/verify the prediction logic and the classification threshold (0.4703) without using a browser, run the following command in PowerShell:

```
$uri = "http://localhost:8080/predict?mean_radius=17.99&mean_texture=10.38&mean_smoothness=0.1184&mean_compactness=0.2776&mean_concavity=0.3001&mean_symmetry=0.2419&mean_fractal_dimension=0.07871&radius_error=1.095&compactness_error=0.04904&concavity_error=0.05373&concave_points_error=0.01587&worst_smoothness=0.1622&worst_compactness=0.6656&worst_concavity=0.7119&worst_concave_points=0.2654&worst_symmetry=0.4601&worst_fractal_dimension=0.1189"

Invoke-RestMethod -Uri $uri -Method Post
```

This should output the following:
`The tumor is classified by the model as: Malignant`