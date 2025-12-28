# Heart Disease Risk Prediction (End-to-End MLOps)

This repository is an end-to-end MLOps project built on the **Heart Disease UCI dataset**.  
It covers the complete lifecycle:

- Dataset download + preprocessing
- Model training (Logistic Regression + Random Forest)
- Evaluation (CV + test metrics + ROC curve)
- Experiment tracking (MLflow)
- Reproducible inference (saved pipeline model)
- Automated tests (Pytest) + CI (GitHub Actions)
- Model-serving API (FastAPI) with `/predict` + `/health`
- Containerization (Docker)
- Deployment (Kubernetes via Docker Desktop)
- Monitoring (Prometheus + Grafana) with `/metrics`

---

## What this project produces

When you run the pipeline you will get:

- `data/raw/heart.csv` (downloaded dataset)
- `artifacts/metrics.json` (training results)
- `artifacts/roc_curve.png` (ROC curve plot)
- `models/best_model.joblib` (final best pipeline model)
- `models/model_meta.json` (metadata about the best model)
- `mlruns/` + `mlflow.db` (MLflow experiment logs)

> Note: In typical production MLOps, **datasets and model binaries are not committed to Git**.  
> This repo generates them via scripts so it works from a clean setup.

---

## Folder structure (important)

```

heart-mlops/
api/
main.py                      # FastAPI server
src/
heart/
data_download.py           # downloads dataset
features.py                # preprocessing + feature pipeline
preprocess_check.py        # sanity check of preprocessing
train.py                   # trains models + logs MLflow + saves outputs
predict_check.py           # loads saved model and runs a sample inference
tests/                         # pytest unit tests
artifacts/                     # metrics + plots (generated)
models/                        # saved model artifacts (generated)
monitoring/
prometheus.yml               # Prometheus scrape config
k8s/
deployment.yaml              # Kubernetes Deployment
service.yaml                 # Kubernetes Service
.github/workflows/ci.yml       # GitHub Actions pipeline
Dockerfile                     # Docker build instructions
docker-compose.monitoring.yml  # Prometheus + Grafana demo stack
requirements.txt               # python dependencies
README.md

````

---

## Prerequisites

### Required
- **Python 3.12+**
- **Git**
- (Optional but recommended) **VS Code**

### For container & deployment tasks
- **Docker Desktop** (Windows)
- Enable **Kubernetes** inside Docker Desktop (for Task 7)

---

## 1) Project setup (Windows)

### 1.1 Clone repo
```cmd
git clone <YOUR_REPO_URL>
cd heart-mlops
````

### 1.2 Create and activate virtual environment

**CMD**

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 1.3 Install dependencies

```cmd
pip install -r requirements.txt
```

### 1.4 Set PYTHONPATH (important for imports)

This project uses imports like `src.heart...`.
So we set the repository root as the Python path.

**CMD**

```cmd
set PYTHONPATH=.
```

**PowerShell**

```powershell
$env:PYTHONPATH="."
```

> If you skip this, you may see errors like: `ModuleNotFoundError: No module named 'src'`.

---

## 2) Download dataset (Task 1)

This script downloads the Heart Disease dataset and saves it to `data/raw/heart.csv`.

```cmd
python src/heart/data_download.py
```

You should see output like:

* Download complete
* Path saved
* Shape: `(303, 14)`
* Target balance counts

---

## 3) Preprocessing sanity check (Task 1–2)

This confirms the preprocessing pipeline fits and transforms correctly.

```cmd
python src/heart/preprocess_check.py
```

Expected output includes something like:

* X shape (303, 13)
* Transformed shape (example: 28 columns after one-hot encoding)

---

## 4) Train models + evaluate (Task 2)

This trains **Logistic Regression** and **Random Forest**, runs:

* Cross-validation metrics
* Test metrics
* ROC curve plot
* Picks best model by test ROC-AUC

Run:

```cmd
python src/heart/train.py
```

Outputs generated:

* `artifacts/metrics.json`
* `artifacts/roc_curve.png`
* `models/best_model.joblib`
* `models/model_meta.json`

---

## 5) Quick inference check (reproducibility proof) (Task 4)

This loads the saved model pipeline and runs a sample prediction.

```cmd
python src/heart/predict_check.py
```

Example output:

* Prediction: 0 or 1
* Confidence: probability value

---

## 6) Experiment tracking (MLflow) (Task 3)

Training logs runs into MLflow.

### 6.1 Start MLflow UI

From repo root:

```cmd
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open in browser:

* [http://127.0.0.1:5000](http://127.0.0.1:5000)

You should see:

* Experiment: `HeartDiseaseRisk`
* Runs for both models
* Metrics, params, artifacts (like ROC plot)

> If MLflow doesn’t open, confirm `mlflow` installed and the command is run inside the venv.

---

## 7) Run the FastAPI server locally (Task 6 base)

### 7.1 Start API

Make sure you already trained the model so `models/best_model.joblib` exists.

```cmd
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 7.2 Test endpoints

**Health**

```cmd
curl http://127.0.0.1:8000/health
```

**Predict**

```cmd
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":55,\"sex\":1,\"cp\":2,\"trestbps\":140,\"chol\":240,\"fbs\":0,\"restecg\":1,\"thalach\":150,\"exang\":0,\"oldpeak\":1.2,\"slope\":1,\"ca\":0,\"thal\":2}"
```

### 7.3 API docs (Swagger)

Open:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 8) Metrics endpoint (Monitoring requirement)

If monitoring code is enabled, check:

```cmd
curl http://127.0.0.1:8000/metrics
```

You should see Prometheus-style metrics like:

* `http_requests_total`
* `http_request_duration_seconds`

---

## 9) Run tests (Task 5)

```cmd
pytest -q
```

Linting:

```cmd
ruff check .
```

---

## 10) CI pipeline (GitHub Actions) (Task 5)

A workflow exists at:

* `.github/workflows/ci.yml`

It performs:

* Install deps
* Lint (ruff)
* Download dataset
* Train model
* Run tests
* Upload artifacts (`artifacts/` and `models/`)

To verify:

* Go to GitHub repo → **Actions** tab → latest workflow run should be green ✅

---

## 11) Docker build + run (Task 6)

### 11.1 Build image

From repo root:

```cmd
docker build -t heart-mlops-api:latest .
```

### 11.2 Run container

```cmd
docker run --rm -p 8000:8000 heart-mlops-api:latest
```

### 11.3 Test container endpoints

In a new terminal:

```cmd
curl http://127.0.0.1:8000/health
```

```cmd
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":55,\"sex\":1,\"cp\":2,\"trestbps\":140,\"chol\":240,\"fbs\":0,\"restecg\":1,\"thalach\":150,\"exang\":0,\"oldpeak\":1.2,\"slope\":1,\"ca\":0,\"thal\":2}"
```

---

## 12) Kubernetes deployment (Docker Desktop Kubernetes) (Task 7)

### 12.1 Enable Kubernetes

Docker Desktop → Settings → Kubernetes → Enable Kubernetes → Apply & Restart

### 12.2 Verify cluster

```cmd
kubectl config use-context docker-desktop
kubectl get nodes
```

### 12.3 Apply manifests

```cmd
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check:

```cmd
kubectl get pods
kubectl get svc
```

### 12.4 Test via port-forward (always works)

Terminal 1:

```cmd
kubectl port-forward svc/heart-api-svc 8000:80
```

Terminal 2:

```cmd
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":55,\"sex\":1,\"cp\":2,\"trestbps\":140,\"chol\":240,\"fbs\":0,\"restecg\":1,\"thalach\":150,\"exang\":0,\"oldpeak\":1.2,\"slope\":1,\"ca\":0,\"thal\":2}"
```

### 12.5 Cleanup k8s

```cmd
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
```

---

## 13) Prometheus + Grafana monitoring stack (Task 8)

This runs:

* heart-api container
* Prometheus scraping `/metrics`
* Grafana dashboard

### 13.1 Start stack

```cmd
docker compose -f docker-compose.monitoring.yml up -d --build
```

Check:

```cmd
docker ps
```

### 13.2 Prometheus UI

Open:

* [http://localhost:9090](http://localhost:9090)

Go to:

* Status → Targets
  You should see `heart-api` as **UP**.

### 13.3 Grafana UI

Open:

* [http://localhost:3000](http://localhost:3000)

Login:

* admin / admin

Add Prometheus datasource:

* URL: `http://prometheus:9090`

Create dashboard query:

* `http_requests_total`

### 13.4 Stop stack

```cmd
docker compose -f docker-compose.monitoring.yml down
```

---

## Troubleshooting (common beginner issues)

### 1) “ModuleNotFoundError: No module named 'src'”

Fix:

* set python path

CMD:

```cmd
set PYTHONPATH=.
```

PowerShell:

```powershell
$env:PYTHONPATH="."
```

### 2) “docker is not recognized”

Docker Desktop is not installed or not running.
Fix:

* install Docker Desktop
* restart terminal
* check: `docker --version`

### 3) “kubectl get nodes” shows localhost:8080 connection refused

Your cluster isn’t running or context is wrong.
Fix:

```cmd
kubectl config get-contexts
kubectl config use-context docker-desktop
kubectl get nodes
```

### 4) Port already in use (8000)

Something else is using port 8000.
Fix:

* stop old uvicorn or container
* or run on different port (example 8001)

### 5) Prometheus target DOWN

Ensure:

* your API exposes `/metrics`
* compose stack is running
* Prometheus target is `heart-api:8000` (container-to-container networking)

---

## Quick command cheat sheet

Setup:

```cmd
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
set PYTHONPATH=.
```

Pipeline:

```cmd
python src/heart/data_download.py
python src/heart/train.py
python src/heart/predict_check.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

API:

```cmd
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Docker:

```cmd
docker build -t heart-mlops-api:latest .
docker run --rm -p 8000:8000 heart-mlops-api:latest
```

Kubernetes:

```cmd
kubectl config use-context docker-desktop
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl port-forward svc/heart-api-svc 8000:80
```

Monitoring stack:

```cmd
docker compose -f docker-compose.monitoring.yml up -d --build
docker compose -f docker-compose.monitoring.yml down
```

---
