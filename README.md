# 🚗 Vehicle Insurance Cross-Sell Prediction — MLOps Project

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen?logo=mongodb)](https://www.mongodb.com/atlas)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20EC2-orange?logo=amazonaws)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [ML Pipeline Architecture](#-ml-pipeline-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [MongoDB Setup](#mongodb-setup)
  - [AWS Setup](#aws-setup)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Deployment on AWS EC2](#-deployment-on-aws-ec2)
- [Author](#-author)

---

## 🔍 Project Overview

This is an **end-to-end MLOps project** built in the **Vehicle Insurance domain**. It predicts whether a health-insured customer would be interested in purchasing **vehicle insurance** from the same company — a classic **cross-sell prediction** problem.

The project follows a production-grade MLOps workflow, covering everything from data ingestion and model training to model evaluation, deployment, and CI/CD automation using **GitHub Actions**, **Docker**, **AWS ECR**, and **AWS EC2**.

---

## 🎯 Problem Statement

An insurance company has customers with existing health insurance policies. The company now wants to predict whether these customers would also be interested in their **Vehicle Insurance** product.

By predicting customer interest, the company can:
- Plan targeted marketing campaigns
- Optimize communication strategies
- Increase revenue through efficient cross-selling

**Target Variable:** `Response` — `1` (Interested) or `0` (Not Interested)

---

## 📊 Dataset Description

The dataset contains customer demographics and policy information:

| Feature | Type | Description |
|---|---|---|
| `id` | int | Unique customer identifier |
| `Gender` | category | Gender of the customer |
| `Age` | int | Age of the customer |
| `Driving_License` | int | 1 if customer has DL, 0 otherwise |
| `Region_Code` | float | Code for the customer's region |
| `Previously_Insured` | int | 1 if already has vehicle insurance |
| `Vehicle_Age` | category | Age of the vehicle |
| `Vehicle_Damage` | category | Whether vehicle was damaged in the past |
| `Annual_Premium` | float | Annual insurance premium amount |
| `Policy_Sales_Channel` | float | Channel through which policy was sold |
| `Vintage` | int | Days customer has been associated with the company |
| `Response` | int | **Target** — 1: Interested, 0: Not Interested |

---

## 🏗️ ML Pipeline Architecture

The project implements a **modular ML pipeline** with the following sequential stages:

```
MongoDB Atlas (Raw Data)
        │
        ▼
┌──────────────────┐
│  Data Ingestion  │  ← Fetches data from MongoDB, performs train-test split
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Data Validation  │  ← Schema checks, null/column validation
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ Data Transformation  │  ← Encoding, scaling, SMOTE for class imbalance
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  Model Trainer   │  ← Random Forest Classifier (entropy, 200 estimators)
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  Model Evaluation    │  ← Compares new vs. production model (S3)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│   Model Pusher       │  ← Pushes accepted model to AWS S3 model registry
└────────┬─────────────┘
         │
         ▼
┌────────────────────────────┐
│  FastAPI Prediction App    │  ← Serves predictions via web interface
└────────────────────────────┘
```

### Key ML Details

- **Algorithm:** Random Forest Classifier
- **Hyperparameters:** `n_estimators=200`, `criterion='entropy'`, `min_samples_split=7`, `min_samples_leaf=6`, `max_depth=10`, `random_state=101`
- **Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique via `imblearn`)
- **Model Registry:** AWS S3 bucket (`my-model-mlopsproj`)
- **Evaluation Threshold:** Model accepted only if improvement > `0.02` (2%) over existing model

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Web Framework** | FastAPI + Uvicorn |
| **Templating** | Jinja2 |
| **Data Processing** | Pandas, NumPy, Scikit-learn, Imbalanced-learn |
| **Database** | MongoDB Atlas (via PyMongo) |
| **Cloud Storage** | AWS S3 (via Boto3) |
| **Serialization** | Dill |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Deployment** | AWS EC2 + AWS ECR |
| **Data Versioning** | DVC |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## 📁 Project Structure

```
MLops-project1---Vehicle-Insurance-Domain/
│
├── .github/
│   └── workflows/
│       └── aws.yaml              # CI/CD GitHub Actions workflow
│
├── config/
│   ├── model.yaml                # Model configuration
│   └── schema.yaml               # Dataset schema for validation
│
├── notebook/
│   ├── mongoDB_demo.ipynb        # MongoDB data push demo
│   └── EDA & Feature Engg.ipynb  # Exploratory Data Analysis
│
├── src/
│   ├── cloud_storage/            # AWS S3 interaction utilities
│   ├── components/
│   │   ├── data_ingestion.py     # Fetch & split data from MongoDB
│   │   ├── data_validation.py    # Schema & data quality checks
│   │   ├── data_transformation.py# Feature engineering & preprocessing
│   │   ├── model_trainer.py      # Model training logic
│   │   ├── model_evaluation.py   # Compare new vs production model
│   │   └── model_pusher.py       # Push model to S3
│   ├── configuration/
│   │   ├── mongo_db_connections.py
│   │   └── aws_connection.py
│   ├── constants/
│   │   └── __init__.py           # All project-wide constants
│   ├── data_access/              # MongoDB data fetch & transform to DataFrame
│   ├── entity/
│   │   ├── config_entity.py      # Config dataclasses for each component
│   │   ├── artifact_entity.py    # Artifact dataclasses for each component
│   │   ├── estimator.py          # Custom estimator wrapper
│   │   └── s3_estimator.py       # S3 model push/pull utilities
│   ├── exception/                # Custom exception handling
│   ├── logger/                   # Custom logging setup
│   ├── pipline/
│   │   ├── training_pipeline.py  # Orchestrates the full training pipeline
│   │   └── prediction_pipeline.py# Handles real-time predictions
│   └── utils/
│       └── main_utils.py         # Shared utility functions
│
├── static/                       # CSS and static assets
├── templates/
│   └── vehicledata.html          # Frontend HTML form for prediction
│
├── app.py                        # FastAPI application entry point
├── demo.py                       # Quick test / demo script
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
├── setup.py                      # Local package setup
├── pyproject.toml                # Build system configuration
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- Conda (recommended) or virtualenv
- MongoDB Atlas account
- AWS account (for S3 model registry and deployment)
- Docker (for containerized deployment)

---

### Environment Setup

**1. Clone the repository**
```bash
git clone https://github.com/Nitesh-raghu45/MLops-project1---Vehicle-Insurance-Domain.git
cd MLops-project1---Vehicle-Insurance-Domain
```

**2. Create and activate a virtual environment**
```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify local package installation**
```bash
pip list | grep src
```

---

### MongoDB Setup

**1.** Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas) and create a new project.

**2.** Create a free **M0 cluster** and set up a **DB user** (username + password).

**3.** Under **Network Access**, add IP `0.0.0.0/0` to allow connections from anywhere.

**4.** Go to your project → **Get Connection String** → **Drivers** → Python 3.6+ → copy the URI and replace `<password>`.

**5.** Set the connection string as an environment variable:

```bash
# PowerShell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>@cluster0.xxx.mongodb.net/"

# Bash / Linux / macOS
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster0.xxx.mongodb.net/"
```

**6.** Use the notebook (`notebook/mongoDB_demo.ipynb`) to push the dataset to MongoDB Atlas.

---

### AWS Setup

**1.** Log into the [AWS Console](https://aws.amazon.com/console/).

**2.** Create an IAM user with **AdministratorAccess** and generate **Access Keys** (CLI).

**3.** Create an **S3 bucket** named `my-model-mlopsproj` in region `us-east-1` (uncheck "Block all public access").

**4.** Set AWS credentials as environment variables:

```bash
# PowerShell
$env:AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"

# Bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
```

---

## ▶️ Running the Application

**Start the FastAPI server locally:**
```bash
python app.py
```

The app will be available at: **http://localhost:5000**

**Trigger model training via API:**
```
GET http://localhost:5000/train
```

**Make a prediction via the web form:**
```
GET/POST http://localhost:5000/
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the vehicle data input form |
| `POST` | `/` | Submits form data and returns insurance interest prediction |
| `GET` | `/train` | Triggers the full ML training pipeline |

### Prediction Output

| Value | Meaning |
|---|---|
| `Response-Yes` | Customer is likely interested in vehicle insurance |
| `Response-No` | Customer is likely **not** interested |

---

## ⚙️ CI/CD Pipeline

The project uses **GitHub Actions** for automated CI/CD. On every push to the main branch:

1. 🐳 **Docker image** is built from the `Dockerfile`
2. 📦 Image is pushed to **AWS ECR** (Elastic Container Registry)
3. 🚀 The updated container is **deployed to AWS EC2** via a self-hosted runner

### GitHub Secrets Required

Add these secrets under **Settings → Secrets and Variables → Actions**:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `AWS_DEFAULT_REGION` | AWS region (e.g., `us-east-1`) |
| `ECR_REPO` | ECR repository URI |

---

## ☁️ Deployment on AWS EC2

**1.** Launch an **EC2 Ubuntu Server 24.04** instance (t2.medium, 30 GB storage).

**2.** SSH into the instance and install Docker:
```bash
sudo apt-get update -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

**3.** Register the EC2 instance as a **GitHub self-hosted runner**:
- Go to your repo → Settings → Actions → Runners → New self-hosted runner
- Follow the Linux download and configure steps on the EC2 instance
- Start the runner with `./run.sh` (or install as a persistent service with `sudo ./svc.sh install && sudo ./svc.sh start`)

**4.** Open port `5080` in EC2 **Security Group** (Inbound rule: Custom TCP, port `5080`, source `0.0.0.0/0`).

**5.** Access your app at:
```
http://<EC2-PUBLIC-IP>:5080
```

**6.** Trigger model training at:
```
http://<EC2-PUBLIC-IP>:5080/train
```

---

## 👤 Author

**Nitesh Raghuwanshi**
- 📧 niteshraghuwanshi68@gmail.com
- 🐙 [GitHub](https://github.com/Nitesh-raghu45)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, please consider giving it a star on GitHub!
