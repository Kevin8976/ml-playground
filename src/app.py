import os

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Playground Inference")

MODEL_PATH = "models/baseline_logreg.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Train it first via src/train_baseline.py"
    )

model = joblib.load(MODEL_PATH)


class Features(BaseModel):
    x: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(f: Features):
    X = np.array([f.x], dtype=float)
    y = int(model.predict(X)[0])
    return {"prediction": y}
