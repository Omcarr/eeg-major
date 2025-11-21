"""
FastAPI WebSocket backend for EEG multi-disorder detection
Now fully compatible with your training script (17 features × 6 channels = 102 features).
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
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

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eeg-server")

# ----------------------------------------------------------
# Modes and Channels
# ----------------------------------------------------------
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

MODEL_DIR = "model"
model_cache = {}
redis_client = None

# ==========================================================
#  MATCHED FEATURE ORDER (17 FEATURES) — EXACT TRAINING ORDER
# ==========================================================
# Your training script sorts feature columns alphabetically.
# After sorting, feature names appear in THIS order.
FEATURE_ORDER = [
    "alpha_power",
    "alpha_relative",
    "beta_power",
    "beta_relative",
    "delta_power",
    "delta_relative",
    "gamma_power",
    "gamma_relative",
    "kurtosis",
    "mean",
    "median",
    "rms",
    "skew",
    "std",
    "theta_power",
    "theta_relative",
    "total_power"
]

# ==========================================================
#  MODEL LOADING
# ==========================================================
def load_model_for_mode(mode: str):
    if mode in model_cache:
        return model_cache[mode]["model"], model_cache[mode]["scaler"]

    model_file = os.path.join(MODEL_DIR, f"{mode}_model.pkl")
    scaler_file = os.path.join(MODEL_DIR, f"{mode}_scaler.pkl")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model missing: {model_file}")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler missing: {scaler_file}")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    model_cache[mode] = {"model": model, "scaler": scaler}
    logger.info(f"✔ Loaded model for mode={mode}")

    return model, scaler

# ==========================================================
#  REDIS EEG BUFFER
# ==========================================================
class RedisEEGBuffer:
    def __init__(self, target_samples=1024):
        self.target_samples = target_samples
        self.channels = DETECTION_MODES[current_mode["mode"]]
        self._lock = asyncio.Lock()

    async def set_mode(self, mode):
        async with self._lock:
            self.channels = DETECTION_MODES[mode]
            logger.info(f"EEG buffer switched to mode {mode}")

    def _key(self, ch):
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
        elif index is not None and index < len(self.channels):
            ch = self.channels[index]
        else:
            return

        key = self._key(ch)
        await redis_client.lpush(key, float(value))
        await redis_client.ltrim(key, 0, self.target_samples - 1)

    async def is_ready(self):
        for ch in self.channels:
            if await redis_client.llen(self._key(ch)) < self.target_samples:
                return False
        return True

    async def get_status(self):
        return {ch: await redis_client.llen(self._key(ch)) for ch in self.channels}

    async def get_data(self):
        data = {}
        for ch in self.channels:
            vals = await redis_client.lrange(self._key(ch), 0, self.target_samples - 1)
            arr = np.array([float(v) for v in reversed(vals)])
            data[ch] = arr
        return data

    async def get_latest_samples(self, count=512):
        data = {}
        for ch in self.channels:
            vals = await redis_client.lrange(self._key(ch), 0, count - 1)
            data[ch] = [float(v) for v in vals]
        return data

    async def clear(self):
        for ch in self.channels:
            await redis_client.delete(self._key(ch))


eeg_buffer = RedisEEGBuffer()

# ==========================================================
#  TRAINING-COMPATIBLE FEATURE EXTRACTOR (17 features)
# ==========================================================
def extract_features(eeg: np.ndarray):
    if eeg.size == 0:
        raise ValueError("Empty EEG array")

    feats = {
        "mean": float(np.mean(eeg)),
        "std": float(np.std(eeg)),
        "skew": float(skew(eeg)),
        "kurtosis": float(kurtosis(eeg)),
        "rms": float(np.sqrt(np.mean(eeg ** 2))),
        "median": float(np.median(eeg)),
    }

    freqs, psd = signal.welch(eeg, fs=128, nperseg=min(128, len(eeg)))
    total_power = float(np.trapz(psd, freqs))
    if total_power <= 0:
        total_power = 1e-12

    feats["total_power"] = total_power

    BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    for band, (lo, hi) in BANDS.items():
        idx = (freqs >= lo) & (freqs <= hi)
        power = float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0
        feats[f"{band}_power"] = power
        feats[f"{band}_relative"] = power / total_power

    return feats

# ==========================================================
#  FEATURE VECTOR (Alphabetical order — EXACT MATCH)
# ==========================================================
def prepare_feature_vector(channel_data, mode):
    feature_map = {}

    for ch in DETECTION_MODES[mode]:
        feats = extract_features(channel_data[ch])
        for f in FEATURE_ORDER:
            feature_map[f"{ch}_{f}"] = feats[f]

    ordered_keys = sorted(feature_map.keys())  # EXACT match to training
    x = np.array([feature_map[k] for k in ordered_keys]).reshape(1, -1)

    return x, ordered_keys

# ==========================================================
#  PREDICTION
# ==========================================================
def predict_mode_from_data(channel_data, mode):
    model, scaler = load_model_for_mode(mode)

    fv, keys = prepare_feature_vector(channel_data, mode)

    expected = scaler.n_features_in_
    if fv.shape[1] != expected:
        raise ValueError(f"Feature mismatch: got {fv.shape[1]}, expected {expected}")

    fv_scaled = scaler.transform(fv)

    pred = int(model.predict(fv_scaled)[0])

    try:
        probs = model.predict_proba(fv_scaled)[0].tolist()
        conf = max(probs)
    except:
        probs, conf = None, None
        
    result= {
        "mode": mode,
        "prediction_raw": pred,
        "class_probabilities": probs,
        "confidence": conf,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channels_used": DETECTION_MODES[mode],
        "feature_count": fv.shape[1]
    }

    print(result)
    return result

# ==========================================================
#  LIFESPAN
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

    await redis_client.ping()
    logger.info("✔ Connected to Redis")

    # clear buffers
    keys = await redis_client.keys("eeg:channel:*")
    if keys:
        await redis_client.delete(*keys)
        logger.info("Cleared old EEG buffer data")

    yield

    await redis_client.close()

# ==========================================================
#  FASTAPI APP
# ==========================================================
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
#  ROUTES
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("eeg_dashboard.html", {"request": request})

@app.get("/api/mode")
async def api_get_mode():
    return {"mode": current_mode["mode"]}

@app.post("/api/set_mode/{mode}")
async def api_set_mode(mode: str):
    if mode not in DETECTION_MODES:
        raise HTTPException(400, "Invalid mode")
    current_mode["mode"] = mode
    await eeg_buffer.set_mode(mode)
    return {"status": "ok"}

@app.get("/api/buffer")
async def api_buffer():
    return {
        "ready": await eeg_buffer.is_ready(),
        "samples": await eeg_buffer.get_status()
    }

@app.get("/api/live_data")
async def api_live(samples: int = 512):
    return {
        "data": await eeg_buffer.get_latest_samples(samples),
        "ready": await eeg_buffer.is_ready(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
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
    try:
        return predict_mode_from_data(data, current_mode["mode"])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction failed: {e}")

# ==========================================================
#  WEBSOCKET (ESP32)
# ==========================================================
@app.websocket("/ws/eeg")
async def ws_eeg(ws: WebSocket):
    await ws.accept()
    logger.info("ESP32 connected")

    try:
        while True:
            msg = await ws.receive_json()

            t = msg.get("type")

            if t == "batch":
                print(msg["data"])
                await eeg_buffer.add_batch(msg["data"])
                ready = await eeg_buffer.is_ready()
                await ws.send_json({"type": "ack", "ready": ready})

            elif t == "sample":
                await eeg_buffer.add_sample(
                    channel=msg.get("channel"),
                    index=msg.get("index"),
                    value=msg["value"]
                )
                await ws.send_json({"type": "ack"})

            elif t == "predict":
                if not await eeg_buffer.is_ready():
                    await ws.send_json({"type": "error", "msg": "Buffer not ready"})
                    continue

                data = await eeg_buffer.get_data()
                try:
                    res = predict_mode_from_data(data, current_mode["mode"])
                    await ws.send_json({"type": "prediction", "result": res})
                except Exception as e:
                    logger.exception("WS prediction error")
                    await ws.send_json({"type": "error", "msg": str(e)})

    except WebSocketDisconnect:
        logger.info("ESP32 disconnected")


# ==========================================================
#  MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
