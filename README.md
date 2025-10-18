## Branches
- main (release)
- dev (active development)

## CI Plan (Week 4)
- GitHub Actions per-branch
- DVC pull (GCS) → tests → quick eval → CML PR comment

## Tests
- Data validation and evaluation tests live in `tests/`.
- Tests read artifacts from env vars:
  - EVAL_DATA_PATH (CSV)
  - MODEL_PATH (joblib/pkl model)
  - MIN_EXPECTED_ACCURACY (default 0.90)
- In CI, we `dvc pull` first and then export these env vars to point to the pulled files.

