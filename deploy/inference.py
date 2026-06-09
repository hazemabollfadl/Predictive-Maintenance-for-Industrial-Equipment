"""
SageMaker inference handler.

SageMaker calls these four functions in order for every request:
  model_fn   → load model + scaler from disk (called once at container start)
  input_fn   → deserialize the raw HTTP body into a Python object
  predict_fn → run the actual forward pass
  output_fn  → serialize the prediction back to an HTTP response
"""

import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definition — must match src/src/models/cnn.py exactly
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, num_sensors: int = 15, window_size: int = 30):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_sensors, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        fc1_input_size = 32 * ((window_size - 2) // 2)
        self.fc1 = nn.Linear(fc1_input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.fc1(self.flatten(x)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# SageMaker contract
# ---------------------------------------------------------------------------

# Must match the window_size used during feature engineering and training.
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 30))


def model_fn(model_dir: str):
    """Load weights and scaler from the unpacked model.tar.gz directory."""
    device = torch.device("cpu")

    model = CNN(num_sensors=15, window_size=WINDOW_SIZE).to(device)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "cnn_FD001.pt"), map_location=device)
    )
    model.eval()

    scaler = joblib.load(os.path.join(model_dir, "scaler_FD001.pkl"))

    return {"model": model, "scaler": scaler, "device": device}


def input_fn(request_body: str, content_type: str = "application/json") -> np.ndarray:
    """Parse the request body into a (WINDOW_SIZE, 15) numpy array.

    Expected JSON shape:
        { "window": [[...15 values...], ...WINDOW_SIZE rows...] }
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)
    window = np.array(payload["window"], dtype=np.float32)

    if window.shape != (WINDOW_SIZE, 15):
        raise ValueError(f"Expected window shape ({WINDOW_SIZE}, 15), got {window.shape}")

    return window


def predict_fn(window: np.ndarray, model_artifacts: dict) -> dict:
    """Scale the input and run a forward pass, returning a raw RUL estimate."""
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    device = model_artifacts["device"]

    scaled = scaler.transform(window)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_rul = model(tensor).item()

    return {
        "predicted_rul_cycles": round(max(predicted_rul, 0), 2),
    }


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """Serialize the prediction dict to a JSON string."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction)
