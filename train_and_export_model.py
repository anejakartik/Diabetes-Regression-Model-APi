import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path(__file__).parent
MODEL_PATH = OUT_DIR / "diabetes_regressor.pth"
SCALER_PATH = OUT_DIR / "scaler_params.json"
METRICS_PATH = OUT_DIR / "model_metrics.json"


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


def train() -> dict:
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    model = DiabetesMLP(input_dim=X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 25
    wait = 0

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = loss_fn(val_preds, y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().ravel()

    y_true = y_test.values
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    torch.save(model.state_dict(), MODEL_PATH)

    scaler_payload = {
        "feature_order": list(X.columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    SCALER_PATH.write_text(json.dumps(scaler_payload, indent=2))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    metrics = train()
    print("Saved model artifacts to:")
    print(f"- {MODEL_PATH}")
    print(f"- {SCALER_PATH}")
    print(f"- {METRICS_PATH}")
    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))
