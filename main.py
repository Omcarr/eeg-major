"""
main.py

FastAPI WebSocket backend for multi-disorder EEG detection
- Uses Redis to store latest 1024 samples per channel
- ESP32 sends data via WebSocket
- Dashboard polls Redis for live data
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Any
from scipy import signal
from scipy.stats import skew, kurtosis
import logging
import uvicorn
from contextlib import asynccontextmanager
import os
import asyncio
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eeg-server")

# -------------------------------------------
# DETECTION MODES
# -------------------------------------------
DETECTION_MODES = {
    "alzheimer": ["Pz", "P3", "P4", "O1", "O2", "Cz"],
    "parkinson": ["Fz", "C3", "C4", "Pz", "T7", "T8"],
    "schizophrenia": ["Fz", "F3", "F4", "Cz", "Pz", "T7"],
    "depression": ["F3", "F4", "Fp1", "Fp2", "Cz", "Pz"],
    "epilepsy": ["F7", "F8", "T7", "T8", "Cz", "Pz"],
    "cognitive": ["Fz", "F3", "F4", "Cz", "Pz", "Oz"],
    "sleep": ["Fz", "Cz", "Pz", "O1", "O2", "T8"]
}

current_mode = {"mode": "alzheimer"}
model_cache: Dict[str, Dict[str, Any]] = {}
MODEL_DIR = "model"

redis_client = None


# -------------------------------------------
# MODEL LOADING
# -------------------------------------------
def load_model_for_mode(mode: str):
    if mode in model_cache:
        return model_cache[mode]["model"], model_cache[mode]["scaler"]

    model_file = os.path.join(MODEL_DIR, f"{mode}_model.pkl")
    scaler_file = os.path.join(MODEL_DIR, f"{mode}_scaler.pkl")

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Missing model for mode '{mode}'.")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    model_cache[mode] = {"model": model, "scaler": scaler}

    logger.info(f"Loaded model for mode={mode}")
    return model, scaler


# -------------------------------------------
# REDIS EEG BUFFER
# -------------------------------------------
class RedisEEGBuffer:
    def __init__(self, target_samples: int = 1024):
        self.target_samples = target_samples
        self.channels = list(DETECTION_MODES[current_mode["mode"]])
        self._lock = asyncio.Lock()

    async def set_mode(self, mode: str):
        async with self._lock:
            self.channels = list(DETECTION_MODES[mode])
            logger.info(f"Buffer switched to mode={mode}")

    def _key(self, ch: str):
        return f"eeg:channel:{ch}"

    async def add_batch(self, batch: Dict[str, List[float]]):
        if not redis_client:
            return

        pipe = redis_client.pipeline()

        for ch, vals in batch.items():
            if ch not in self.channels:
                continue
            key = self._key(ch)

            for v in vals:
                pipe.lpush(key, float(v))
            pipe.ltrim(key, 0, self.target_samples - 1)

        await pipe.execute()

    async def add_sample(self, *, channel=None, index=None, value=None):
        if not redis_client:
            return

        if channel:
            ch = channel
        elif index is not None:
            if 0 <= index < len(self.channels):
                ch = self.channels[index]
            else:
                return
        else:
            return

        key = self._key(ch)
        await redis_client.lpush(key, float(value))
        await redis_client.ltrim(key, 0, self.target_samples - 1)

    async def is_ready(self):
        for ch in self.channels:
            key = self._key(ch)
            if await redis_client.llen(key) < self.target_samples:
                return False
        return True

    async def get_status(self):
        status = {}
        for ch in self.channels:
            status[ch] = await redis_client.llen(self._key(ch))
        return status

    async def get_data(self):
        data = {}
        for ch in self.channels:
            values = await redis_client.lrange(self._key(ch), 0, self.target_samples - 1)
            arr = np.array([float(v) for v in reversed(values)])
            data[ch] = arr
        return data

    async def get_latest_samples(self, count=512):
        data = {}
        for ch in self.channels:
            values = await redis_client.lrange(self._key(ch), 0, count - 1)
            data[ch] = [float(v) for v in values]
        return data

    async def clear(self):
        for ch in self.channels:
            await redis_client.delete(self._key(ch))


eeg_buffer = RedisEEGBuffer(target_samples=1024)


# -------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------
def extract_features(signal_data: np.ndarray):
    feats = {
        "mean": float(np.mean(signal_data)),
        "std": float(np.std(signal_data)),
        "skew": float(skew(signal_data)),
        "kurtosis": float(kurtosis(signal_data)),
        "rms": float(np.sqrt(np.mean(signal_data ** 2))),
        "median": float(np.median(signal_data)),
    }

    freqs, psd = signal.welch(signal_data, fs=128, nperseg=128)
    total_power = max(float(np.trapz(psd, freqs)), 1e-12)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    for name, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        power = float(np.trapz(psd[idx], freqs[idx]))
        feats[f"{name}_power"] = power
        feats[f"{name}_relative"] = power / total_power

    return feats


# -------------------------------------------
# FEATURE VECTORIZATION
# -------------------------------------------
def prepare_feature_vector(channel_data, mode):
    expected = DETECTION_MODES[mode]

    feature_map = {}
    for ch in expected:
        feats = extract_features(channel_data[ch])
        for f, v in feats.items():
            feature_map[f"{ch}_{f}"] = v

    keys = sorted(feature_map)
    x = np.array([feature_map[k] for k in keys]).reshape(1, -1)
    return x, keys


def predict_mode_from_data(channel_data, mode):
    model, scaler = load_model_for_mode(mode)
    fv, keys = prepare_feature_vector(channel_data, mode)
    fv_scaled = scaler.transform(fv)
    pred = model.predict(fv_scaled)[0]

    probs = None
    try:
        probs = model.predict_proba(fv_scaled)[0].tolist()
    except:
        pass

    return {
        "mode": mode,
        "prediction_raw": int(pred),
        "class_probabilities": probs,
        "confidence": max(probs) if probs else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channels_used": DETECTION_MODES[mode],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client

    logger.info("Starting EEG API with Redis")

    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

    await redis_client.ping()
    logger.info("Connected to Redis")

    # CLEAR EEG keys on startup
    keys = await redis_client.keys("eeg:channel:*")
    if keys:
        await redis_client.delete(*keys)
        logger.info("Cleared EEG buffers on startup")

    # Preload model
    try:
        load_model_for_mode(current_mode["mode"])
    except Exception as e:
        logger.warning(f"Model preload failed: {e}")

    yield  # REQUIRED

    # Cleanup
    await redis_client.close()
    logger.info("Redis connection closed")


# -------------------------------------------
# FASTAPI APP
# -------------------------------------------
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------
# HTTP ROUTES
# -------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("eeg_dashboard.html", {"request": request})


@app.get("/api/mode")
async def api_get_mode():
    return {"mode": current_mode["mode"], "channels": DETECTION_MODES[current_mode["mode"]]}


@app.post("/api/set_mode/{mode}")
async def api_set_mode(mode: str):
    mode = mode.lower()
    if mode not in DETECTION_MODES:
        raise HTTPException(400, f"Unknown mode '{mode}'")

    current_mode["mode"] = mode
    await eeg_buffer.set_mode(mode)
    return {"status": "ok", "mode": mode}


@app.get("/api/buffer")
async def api_get_buffer():
    return {"ready": await eeg_buffer.is_ready(), "samples": await eeg_buffer.get_status()}


@app.get("/api/live_data")
async def api_live(samples: int = 512):
    return {
        "data": await eeg_buffer.get_latest_samples(samples),
        "ready": await eeg_buffer.is_ready(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/clear")
async def api_clear():
    await eeg_buffer.clear()
    return {"status": "cleared"}


@app.post("/api/predict")
async def api_predict():
    if not await eeg_buffer.is_ready():
        raise HTTPException(400, "Buffer not ready")

    data = await eeg_buffer.get_data()
    return predict_mode_from_data(data, current_mode["mode"])


# -------------------------------------------
# WEBSOCKET FOR ESP32
# -------------------------------------------
@app.websocket("/ws/eeg")
async def ws_eeg(ws: WebSocket):
    await ws.accept()
    logger.info("ESP32 connected")

    try:
        while True:
            msg = await ws.receive_json()

            if msg.get("type") == "batch":
                print(msg["data"])
                await eeg_buffer.add_batch(msg["data"])
                ready = await eeg_buffer.is_ready()
                await ws.send_json({"type": "ack", "ready": ready})

            elif msg.get("type") == "sample":
                await eeg_buffer.add_sample(channel=msg.get("channel"), index=msg.get("index"), value=msg["value"])
                await ws.send_json({"type": "ack"})

            elif msg.get("type") == "predict":
                if not await eeg_buffer.is_ready():
                    await ws.send_json({"type": "error", "msg": "Buffer not ready"})
                    continue
                data = await eeg_buffer.get_data()
                result = predict_mode_from_data(data, current_mode["mode"])
                await ws.send_json({"type": "prediction", "result": result})

    except WebSocketDisconnect:
        logger.info("ESP32 disconnected")


# -------------------------------------------
# MAIN ENTRY
# -------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
