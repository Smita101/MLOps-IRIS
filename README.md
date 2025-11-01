# IRIS MLOps Pipeline – CI/CD Implementation

## Branches
- **main** – stable / release branch (protected)
- **dev** – active development branch (feature integration)
- **feat/ga6-api** – feature branch for Week 6 Continuous Deployment (Docker + Kubernetes + GitHub Actions)

---

## Continuous Integration Setup (Week 4)
- GitHub Actions configured per branch (`ci-dev.yml`, `ci-main.yml`)
- Each workflow performs:
  1. Checkout + Python setup  
  2. `dvc pull` from GCS (remote from Week 2)  
  3. Data-validation and evaluation tests (`pytest`)  
  4. Quick evaluation → generate/update `metrics.json`  
  5. Automated **CML PR comment** posting the results  
- The pipeline demonstrates end-to-end MLOps integration with **DVC + CI + CML**.

---

## Continuous Deployment Setup (Week 6)
This stage builds on top of the CI pipeline to achieve complete **CI/CD automation**.

- **GitHub Actions** workflow (`.github/workflows/cd.yml`) automatically:
  1. Builds the **Docker image** for the IRIS FastAPI application using the `Dockerfile`.  
  2. Pushes the image to **Google Artifact Registry (GAR)**.  
  3. Deploys the containerized app to **Google Kubernetes Engine (GKE)** using Kubernetes manifests  
     (`k8s/deployment.yaml` and `k8s/service.yaml`).  
- The **GKE Service** exposes the API through a **LoadBalancer**, making it accessible via an external IP.  

### Endpoints
- **`/health`** → Returns application health status.  
- **`/predict`** → Accepts IRIS flower measurements and returns the predicted class.  

### Tech Stack
- **FastAPI** – API framework  
- **Docker** – Containerization  
- **GitHub Actions** – CI/CD automation  
- **Google Artifact Registry (GAR)** – Image storage  
- **Google Kubernetes Engine (GKE)** – Deployment and orchestration  

---

## Tests
- Data-validation and evaluation tests live under the `tests/` folder.
- Tests use environment variables:
  - `EVAL_DATA_PATH` – evaluation CSV  
  - `MODEL_PATH` – trained model artifact  
  - `MIN_EXPECTED_ACCURACY` – default 0.90  
- In CI, these are exported after `dvc pull` to point to pulled files.

---

## Learnings
- How to containerize a machine learning API using **Docker** and **FastAPI**.  
- How to push Docker images securely to **Google Artifact Registry** using service accounts.  
- How to use **GitHub Actions** for automated build, push, and deploy to **Kubernetes**.  
- How **Deployments** and **Services** work together to expose containerized applications.  
- How to differentiate between **Pods** (deployable units) and **Containers** (runtime instances).  
- How to verify deployments via public LoadBalancer endpoints and ensure reproducible API responses.

---
