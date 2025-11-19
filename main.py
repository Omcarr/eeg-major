"""
FastAPI WebSocket Server for Real-time EEG Alzheimer's Detection
Receives data from ESP32, visualizes EEG waves, and makes predictions
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List
from collections import deque
from scipy import signal
from scipy.stats import skew, kurtosis
import logging
import uvicorn
from contextlib import asynccontextmanager

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + Shutdown"""
    logger.info("Starting EEG Alzheimer's Detection API...")

    try:
        load_model()
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.warning(f"Model NOT loaded: {e}")

    yield

    logger.info("Shutting down API...")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EEGDataBuffer:
    """Thread-safe buffer for storing EEG data from six channels"""

    def __init__(self, target_samples=1024):
        self.target_samples = target_samples
        self.channels = ['Pz', 'P3', 'P4', 'O1', 'O2', 'Cz']
        self.buffers = {ch: deque(maxlen=target_samples) for ch in self.channels}
        self.is_ready = False

    def add_sample(self, channel: str, value: float):
        if channel in self.buffers:
            self.buffers[channel].append(float(value))
            self._update_ready()

    def add_batch(self, batch: Dict[str, List[float]]):
        for ch, values in batch.items():
            if ch in self.buffers:
                for v in values:
                    self.buffers[ch].append(float(v))
        self._update_ready()

    def _update_ready(self):
        self.is_ready = all(len(buf) >= self.target_samples for buf in self.buffers.values())

    def get_data(self) -> Dict[str, np.ndarray]:
        return {ch: np.array(buf) for ch, buf in self.buffers.items()}

    def clear(self):
        for buf in self.buffers.values():
            buf.clear()
        self.is_ready = False

    def get_status(self):
        return {ch: len(buf) for ch, buf in self.buffers.items()}

eeg_buffer = EEGDataBuffer()

# ============================================
# MODEL LOADING
# ============================================

model_cache = {"model": None, "scaler": None, "loaded": False}

def load_model():
    if not model_cache["loaded"]:
        model_cache["model"] = joblib.load("model/eeg_alzheimer_model.pkl")
        model_cache["scaler"] = joblib.load("model/eeg_scaler.pkl")
        model_cache["loaded"] = True

    return model_cache["model"], model_cache["scaler"]

def extract_features(eeg_signal: np.ndarray):
    features = {}

    # Time-domain
    features["mean"] = float(np.mean(eeg_signal))
    features["std"] = float(np.std(eeg_signal))
    features["skew"] = float(skew(eeg_signal))
    features["kurtosis"] = float(kurtosis(eeg_signal))
    features["rms"] = float(np.sqrt(np.mean(eeg_signal**2)))
    features["median"] = float(np.median(eeg_signal))

    # Frequency-domain
    freqs, psd = signal.welch(eeg_signal, fs=128, nperseg=128)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }

    total_power = float(np.trapz(psd, freqs))
    features["total_power"] = total_power

    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        power = float(np.trapz(psd[idx], freqs[idx]))
        features[f"{band}_power"] = power
        features[f"{band}_relative"] = power / (total_power + 1e-10)

    return features


def prepare_feature_vector(channel_data):
    feature_dict = {}

    for ch, signal_data in channel_data.items():
        feats = extract_features(signal_data)
        for name, val in feats.items():
            feature_dict[f"{ch}_{name}"] = val

    # sort keys to match model training order
    keys = sorted(feature_dict.keys())
    vector = np.array([feature_dict[k] for k in keys]).reshape(1, -1)
    return vector


def make_prediction(channel_data):
    model, scaler = load_model()
    features = prepare_feature_vector(channel_data)
    scaled = scaler.transform(features)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    return {
        "prediction": "Alzheimer's Disease" if pred == 1 else "Healthy",
        "confidence": float(max(prob)),
        "alzheimer_probability": float(prob[1]),
        "healthy_probability": float(prob[0]),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("eeg_dashboard.html", {"request": request})


@app.get("/api/buffer")
async def buffer_status():
    return {
        "ready": eeg_buffer.is_ready,
        "samples": eeg_buffer.get_status()
    }


@app.post("/api/predict")
async def api_predict():
    if not eeg_buffer.is_ready:
        raise HTTPException(status_code=400, detail="Not enough data in buffer")

    data = eeg_buffer.get_data()
    return make_prediction(data)


@app.post("/api/clear")
async def clear():
    eeg_buffer.clear()
    return {"status": "cleared"}


@app.websocket("/ws/eeg")
async def ws_eeg(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "sample":
                eeg_buffer.add_sample(msg["channel"], msg["value"])

            elif mtype == "batch":
                print(msg["data"])
                eeg_buffer.add_batch(msg["data"])

            elif mtype == "predict":
                if eeg_buffer.is_ready:
                    result = make_prediction(eeg_buffer.get_data())
                    await ws.send_json({"type": "prediction", "result": result})
                else:
                    await ws.send_json({"type": "error", "msg": "Buffer not ready"})

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)






