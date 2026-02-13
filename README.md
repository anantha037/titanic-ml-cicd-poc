# Titanic ML CI/CD Deployment POC

This repository demonstrates a minimal Proof-of-Concept (POC) for
deploying a Machine Learning model using CI/CD practices.\
The focus of this project is not model complexity, but understanding the
end-to-end ML deployment lifecycle including training, versioning,
deployment, monitoring, and rollback strategies.

------------------------------------------------------------------------

## 🚀 Project Overview

This project builds a Titanic survival prediction API using FastAPI and
integrates:

-   Automated model training
-   Model versioning
-   CI/CD with GitHub Actions
-   Logging and monitoring
-   Manual rollback strategy

The goal is to showcase how machine learning systems move from training
to deployment in a structured engineering workflow.

------------------------------------------------------------------------

## 🧠 Architecture Diagram

``` mermaid
flowchart LR
    A[GitHub Push] --> B[GitHub Actions CI/CD]
    B --> C[Train.py - Model Training]
    C --> D[Versioned Model Files]
    D --> E[FastAPI Deployment]
    E --> F[Prediction API]
    E --> G[/metrics Monitoring Endpoint]
```

This diagram shows how code changes trigger CI/CD, which trains a
versioned model and validates the deployment API.

------------------------------------------------------------------------

## 📁 Project Structure

    titanic-ml-cicd-poc/
    │
    ├── data/                  # Titanic dataset
    ├── models/                # Saved models (ignored in git)
    ├── train.py               # Training pipeline
    ├── app.py                 # FastAPI deployment API
    ├── requirements.txt
    └── .github/workflows/     # CI/CD pipelines

------------------------------------------------------------------------

## ⚙️ Model Training

The model is trained using Logistic Regression on selected Titanic
features:

-   Pclass
-   Sex
-   Age
-   SibSp
-   Parch
-   Fare
-   Embarked

Run training locally:

    python train.py v1

This generates:

    models/model_v1.joblib

------------------------------------------------------------------------

## 🌐 API Deployment

Start the API locally:

    MODEL_VERSION=v1 uvicorn app:app --reload

Available endpoints:

-   `/` → Health check
-   `/predict` → Survival prediction
-   `/metrics` → Monitoring metrics

Example prediction request:

``` json
{
  "Pclass": 3,
  "Sex": 0,
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": 0
}
```

------------------------------------------------------------------------

## 📊 Monitoring & Logging

The API exposes Prometheus-compatible metrics including:

-   Request count
-   Prediction latency

Metrics endpoint:

    /metrics

Logging is implemented using Python's logging module to track prediction
requests and results.

------------------------------------------------------------------------

## 🔁 CI/CD Pipeline

GitHub Actions automatically:

1.  Installs dependencies
2.  Trains a new model version
3.  Loads the versioned model
4.  Validates FastAPI deployment

Each pipeline run creates a new model version:

    model_v1
    model_v2
    model_v3

------------------------------------------------------------------------

## 🔄 Model Versioning & Rollback

The active model is controlled using environment variables.

Example rollback:

    MODEL_VERSION=v2 uvicorn app:app

This allows switching to a previous stable model without retraining or
modifying code.

------------------------------------------------------------------------

## 🎯 Purpose of This Project

This repository is a learning-focused POC designed to demonstrate:

-   CI/CD concepts for Machine Learning
-   Deployment validation strategies
-   Monitoring and logging practices
-   Versioned model management

------------------------------------------------------------------------

## 📌 Future Improvements

-   Model registry integration
-   Automated data validation
-   Performance evaluation in CI/CD
-   Cloud deployment integration
