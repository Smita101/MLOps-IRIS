## Branches
- **main** – stable / release branch (protected)
- **dev** – active development branch (feature integration)

## Continuous Integration Setup (Week 4)
- GitHub Actions configured per branch (`ci-dev.yml`, `ci-main.yml`)
- Each workflow performs:
  1. Checkout + Python setup  
  2. `dvc pull` from GCS (remote from Week 2)  
  3. Data-validation and evaluation tests (`pytest`)  
  4. Quick evaluation → generate/update `metrics.json`  
  5. Automated **CML PR comment** posting the results  
- The pipeline demonstrates end-to-end MLOps integration with DVC + CI + CML.

## Tests
- Data-validation and evaluation tests live under `tests/`.
- Tests use environment variables:
  - `EVAL_DATA_PATH` – evaluation CSV  
  - `MODEL_PATH` – trained model artifact  
  - `MIN_EXPECTED_ACCURACY` – default 0.90  
- In CI, these are exported after `dvc pull` to point to pulled files.


