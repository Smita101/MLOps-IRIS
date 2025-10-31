# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import os, joblib, numpy as np

app = FastAPI(title="IRIS API")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: IrisFeatures):
    X = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]])
    y = load_model().predict(X).tolist()[0]
    return {"prediction": y}
