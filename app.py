import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "diabetes_regressor.pth"
SCALER_PATH = BASE_DIR / "scaler_params.json"
METRICS_PATH = BASE_DIR / "model_metrics.json"


class DiabetesMLP(nn.Module):
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiabetesInput(BaseModel):
    age: float = Field(..., description="Standardized age feature")
    sex: float = Field(..., description="Standardized sex feature")
    bmi: float = Field(..., description="Standardized BMI feature")
    bp: float = Field(..., description="Standardized blood pressure feature")
    s1: float = Field(..., description="Standardized S1 blood serum measurement")
    s2: float = Field(..., description="Standardized S2 blood serum measurement")
    s3: float = Field(..., description="Standardized S3 blood serum measurement")
    s4: float = Field(..., description="Standardized S4 blood serum measurement")
    s5: float = Field(..., description="Standardized S5 blood serum measurement")
    s6: float = Field(..., description="Standardized S6 blood serum measurement")


def load_scaler() -> Dict:
    if not SCALER_PATH.exists():
        raise FileNotFoundError("scaler_params.json not found. Run train_and_export_model.py first.")
    return json.loads(SCALER_PATH.read_text())


def load_metrics() -> Dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


scaler_payload = load_scaler()
feature_order = scaler_payload["feature_order"]
mean = np.array(scaler_payload["mean"], dtype=np.float32)
scale = np.array(scaler_payload["scale"], dtype=np.float32)

model = DiabetesMLP(input_dim=len(feature_order))
if not MODEL_PATH.exists():
    raise FileNotFoundError("diabetes_regressor.pth not found. Run train_and_export_model.py first.")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

app = FastAPI(
    title="EAI6010 Diabetes Progression Predictor",
    description="Microservice exposing the off-the-shelf adapted PyTorch regression model.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Service is running",
        "docs": "/docs",
        "predict_endpoint": "/predict",
        "feature_order": feature_order,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "metrics": load_metrics(),
    }


@app.post("/predict")
def predict(payload: DiabetesInput):
    raw = np.array([[getattr(payload, f) for f in feature_order]], dtype=np.float32)
    x_scaled = (raw - mean) / scale

    with torch.no_grad():
        pred = model(torch.tensor(x_scaled, dtype=torch.float32)).item()

    return {
        "input": payload.model_dump(),
        "predicted_disease_progression": round(float(pred), 3),
        "units": "quantitative progression score (higher means worse progression)",
    }
