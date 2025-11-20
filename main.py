"""
main.py

FastAPI WebSocket backend for multi-disorder EEG detection
- Accepts ESP batches like:
  { "type": "batch", "data": { "Pz":[...], "P3":[...], ... } }
- Accepts single samples like:
  { "type": "sample", "channel": "Pz", "value": 1.23 }
  or
  { "type": "sample", "index": 0, "value": 1.23 }  # index maps to current mode channels 0..5
- Mode switching via POST /api/set_mode/{mode}
- Prediction via POST /api/predict or ws { "type": "predict" }
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Any
from collections import deque
from scipy import signal
from scipy.stats import skew, kurtosis
import logging
import uvicorn
from contextlib import asynccontextmanager
import os
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eeg-server")

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


model_cache: Dict[str, Dict[str, Any]] = {}  # mode -> {"model":..., "scaler":...}
MODEL_DIR = "model"

def load_model_for_mode(mode: str):
    if mode in model_cache:
        return model_cache[mode]["model"], model_cache[mode]["scaler"]

    model_file = os.path.join(MODEL_DIR, f"{mode}_model.pkl")
    scaler_file = os.path.join(MODEL_DIR, f"{mode}_scaler.pkl")

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Missing model/scaler for mode '{mode}'. "
                                f"Expected {model_file} and {scaler_file}")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    model_cache[mode] = {"model": model, "scaler": scaler}
    logger.info(f"Loaded model for mode={mode}")
    return model, scaler

# -------------------------
# EEG Buffer (thread-safe)
# -------------------------
class EEGDataBuffer:
    def __init__(self, target_samples: int = 1024):
        self.target_samples = target_samples
        self.channels = list(DETECTION_MODES[current_mode["mode"]])  # current expected channels
        self.buffers: Dict[str, deque] = {ch: deque(maxlen=self.target_samples) for ch in self.channels}
        self._lock = asyncio.Lock()

    async def set_mode(self, mode: str):
        async with self._lock:
            self.channels = list(DETECTION_MODES[mode])
            self.buffers = {ch: deque(maxlen=self.target_samples) for ch in self.channels}
            logger.info(f"Buffer mode set to {mode}, channels={self.channels}")

    async def add_batch(self, batch: Dict[str, List[float]]):
        """
        Accepts {'Pz':[...], 'P3':[...], ...}. Only append samples for channels present
        in the current mode's expected channels. Non-listed channels are ignored.
        """
        async with self._lock:
            for ch, vals in batch.items():
                if ch not in self.buffers:
                    # ignore channels not used by current mode
                    continue
                # append sequentially; if vals is scalar mistakenly, handle gracefully
                if not isinstance(vals, (list, tuple, np.ndarray)):
                    try:
                        vals = [float(vals)]
                    except Exception:
                        continue
                for v in vals:
                    try:
                        self.buffers[ch].append(float(v))
                    except Exception:
                        # ignore malformed entries
                        continue
            # update readiness
            # do NOT set ready true unless ALL channels have >= target_samples
            # (this ensures prepare_feature_vector can rely on complete data)
            # readiness updated in method below if needed

    async def add_sample(self, *, channel: str = None, index: int = None, value: float = None):
        """
        Add a single sample. Accepts channel name OR index (0..5 refers to current channel order).
        """
        async with self._lock:
            ch = None
            if channel:
                ch = channel
            elif index is not None:
                if 0 <= index < len(self.channels):
                    ch = self.channels[index]
            if ch and ch in self.buffers:
                try:
                    self.buffers[ch].append(float(value))
                except Exception:
                    pass

    async def is_ready(self) -> bool:
        async with self._lock:
            return all(len(buf) >= self.target_samples for buf in self.buffers.values())

    async def get_status(self) -> Dict[str, int]:
        async with self._lock:
            return {ch: len(buf) for ch, buf in self.buffers.items()}

    async def get_data(self) -> Dict[str, np.ndarray]:
        async with self._lock:
            # Return np arrays (copy)
            return {ch: np.array(list(buf)) for ch, buf in self.buffers.items()}

    async def clear(self):
        async with self._lock:
            for buf in self.buffers.values():
                buf.clear()

# instantiate buffer
eeg_buffer = EEGDataBuffer(target_samples=1024)


def extract_features(eeg_signal: np.ndarray):
    """
    Returns a dict of features for a single 1-D numpy array signal.
    """
    if eeg_signal.size == 0:
        raise ValueError("Empty signal passed to extract_features")

    feats: Dict[str, float] = {}
    feats["mean"] = float(np.mean(eeg_signal))
    feats["std"] = float(np.std(eeg_signal))
    feats["skew"] = float(skew(eeg_signal))
    feats["kurtosis"] = float(kurtosis(eeg_signal))
    feats["rms"] = float(np.sqrt(np.mean(eeg_signal**2)))
    feats["median"] = float(np.median(eeg_signal))

    # PSD features (Welch)
    freqs, psd = signal.welch(eeg_signal, fs=128, nperseg=128)
    total_power = float(np.trapz(psd, freqs))
    feats["total_power"] = total_power if total_power > 0 else 1e-12

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }

    for bname, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        power = float(np.trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0
        feats[f"{bname}_power"] = power
        feats[f"{bname}_relative"] = power / (total_power + 1e-12)

    return feats

def prepare_feature_vector(channel_data: Dict[str, np.ndarray], mode: str):
    """
    channel_data must contain EXACTLY the channels in DETECTION_MODES[mode] and each
    array must have at least target_samples length.
    Returns 2D numpy array (1, n_features) with sorted feature keys for deterministic order.
    """
    expected = DETECTION_MODES[mode]

    # validate presence and lengths
    for ch in expected:
        if ch not in channel_data:
            raise ValueError(f"Missing channel '{ch}' required for mode '{mode}'")
        if channel_data[ch].size < eeg_buffer.target_samples:
            raise ValueError(f"Channel '{ch}' has insufficient samples ({channel_data[ch].size})")

    feature_map: Dict[str, float] = {}
    for ch in expected:
        feats = extract_features(channel_data[ch])
        for fname, val in feats.items():
            feature_map[f"{ch}_{fname}"] = float(val)

    # deterministic ordering
    keys = sorted(feature_map.keys())
    vector = np.array([feature_map[k] for k in keys]).reshape(1, -1)
    return vector, keys  # return keys for debugging if needed

def predict_mode_from_data(channel_data: Dict[str, np.ndarray], mode: str):
    """
    Prepare features, scale and predict using mode-specific model.
    Returns JSON-serializable dict with predictions and probs.
    """
    model, scaler = load_model_for_mode(mode)
    fv, feature_keys = prepare_feature_vector(channel_data, mode)
    # check scaler shape compatibility (best-effort)
    try:
        fv_scaled = scaler.transform(fv)
    except Exception as e:
        raise RuntimeError(f"Scaler transform failed: {e}")

    preds = model.predict(fv_scaled)
    probs = None
    try:
        probs = model.predict_proba(fv_scaled)[0].tolist()
    except Exception:
        # model might not support predict_proba
        probs = None

    pred = preds[0]
    response = {
        "mode": mode,
        "prediction_raw": int(pred) if isinstance(pred, (np.integer, int)) else str(pred),
        "class_probabilities": probs,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "channels_used": DETECTION_MODES[mode],
    }

    # attempt to include human-friendly label if model has classes_
    if hasattr(model, "classes_"):
        try:
            classes = [str(c) for c in model.classes_]
            response["classes"] = classes
        except Exception:
            pass

    # confidence as max prob if available
    if probs:
        response["confidence"] = float(max(probs))
    else:
        response["confidence"] = None

    return response

# -------------------------
# FastAPI app
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EEG API")
    # optionally preload models for default mode (non-blocking)
    try:
        load_model_for_mode(current_mode["mode"])
    except Exception as e:
        logger.warning(f"Could not preload model for default mode: {e}")
    yield
    logger.info("Shutting down EEG API")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -------------------------
# HTTP endpoints
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("eeg_dashboard.html", {"request": request})

@app.get("/api/mode")
async def api_get_mode():
    mode = current_mode["mode"]
    return {"mode": mode, "channels": DETECTION_MODES[mode]}

@app.post("/api/set_mode/{mode}")
async def api_set_mode(mode: str):
    mode = mode.lower()
    if mode not in DETECTION_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'")
    current_mode["mode"] = mode
    await eeg_buffer.set_mode(mode)
    # attempt to (lazy) load model so errors surface early
    try:
        load_model_for_mode(mode)
    except Exception as e:
        logger.warning(f"Model load warning for mode={mode}: {e}")
    return {"status": "ok", "mode": mode, "channels": DETECTION_MODES[mode]}

@app.get("/api/buffer")
async def api_get_buffer():
    ready = await eeg_buffer.is_ready()
    status = await eeg_buffer.get_status()
    return {"ready": ready, "samples": status}

@app.post("/api/clear")
async def api_clear():
    await eeg_buffer.clear()
    return {"status": "cleared"}

@app.post("/api/predict")
async def api_predict():
    ready = await eeg_buffer.is_ready()
    if not ready:
        raise HTTPException(status_code=400, detail="Buffer not ready (need full 1024 samples per channel)")
    data = await eeg_buffer.get_data()
    mode = current_mode["mode"]
    try:
        res = predict_mode_from_data(data, mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(res)


@app.websocket("/ws/eeg")
async def ws_eeg(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except Exception:
                # Non-json or disconnected
                break

            # msg expected: {"type":"batch", "data":{...}} or {"type":"sample", "channel":.., "value":..}
            mtype = msg.get("type")
            if mtype == "batch":
                data = msg.get("data")
                print(data)
                if not isinstance(data, dict):
                    await ws.send_json({"type": "error", "msg": "Invalid batch: 'data' must be object"})
                    continue
                await eeg_buffer.add_batch(data)
                # optionally send buffer status
                ready = await eeg_buffer.is_ready()
                await ws.send_json({"type": "buffer", "ready": ready})
            elif mtype == "sample":
                # support 'channel' or 'index'
                if "channel" in msg:
                    await eeg_buffer.add_sample(channel=msg.get("channel"), value=msg.get("value"))
                else:
                    await eeg_buffer.add_sample(index=msg.get("index"), value=msg.get("value"))
                ready = await eeg_buffer.is_ready()
                # do not spam - only send status every time for now
                await ws.send_json({"type": "buffer", "ready": ready})
            elif mtype == "predict":
                ready = await eeg_buffer.is_ready()
                if not ready:
                    await ws.send_json({"type": "error", "msg": "Buffer not ready"})
                    continue
                data = await eeg_buffer.get_data()
                mode = current_mode["mode"]
                try:
                    res = predict_mode_from_data(data, mode)
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": f"Prediction failed: {e}"})
                    continue
                await ws.send_json({"type": "prediction", "result": res})
            else:
                await ws.send_json({"type": "error", "msg": "Unknown message type"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket loop error: %s", e)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
