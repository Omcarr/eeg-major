# ui.py
"""
Classic EEG stacked Streamlit dashboard (single file).
Features:
 - Classic vertical stacked EEG waveforms (auto-scaling)
 - Live ingestion via backend WebSocket (/ws/eeg) (requires websocket-client)
 - Demo-mode simulated EEG if WS not available
 - Band-power cards (delta/theta/alpha/beta/gamma)
 - Prediction panel (calls /api/predict or receives WS predictions)
 - Uses Streamlit >=1.40 APIs (st.rerun, width="stretch")
"""

from collections import deque
import json
import math
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import signal
from scipy.stats import kurtosis, skew

# Optional websocket-client import
try:
    import websocket  # websocket-client package name
    WEBSOCKET_AVAILABLE = True
except Exception:
    WEBSOCKET_AVAILABLE = False

API_BASE = os.environ.get("EEG_API_BASE", "http://localhost:8000")
WS_BASE = os.environ.get("EEG_WS_BASE", "ws://localhost:8000/ws/eeg")

import requests

GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-1.5-flash"

def gemini_interpret(pred_label, conf, band_summary):
    """
    Generate a short, formatted interpretation using Gemini.
    Keeps output extremely small and clinician-style.
    """
    prompt = f"""
You are an EEG analysis assistant.

Prediction: {pred_label}
Confidence: {conf:.2f}
Band summary: {band_summary}

Write a VERY SHORT interpretation (2–3 lines), concise, professional, no filler.
Format:
- Insight 1
- Insight 2
"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts":[{"text": prompt}]}]}

        res = requests.post(url, headers=headers, json=payload, timeout=5)
        data = res.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except Exception as e:
        return f"- Gemini interpretation unavailable\n- Error: {e}"














SAMPLE_RATE = 128
FFT_SIZE = 256
DISPLAY_BUFFER = 2048   # how many samples we keep per channel
PLOT_WINDOW = 512       # how many points to draw per channel (scroll width)
REFRESH_DEFAULT = 0.5   # seconds

DETECTION_MODES = {
    "alzheimer": ["Pz", "P3", "P4", "O1", "O2", "Cz"],
    "parkinson": ["Fz", "C3", "C4", "Pz", "T7", "T8"],
    "schizophrenia": ["Fz", "F3", "F4", "Cz", "Pz", "T7"],
    "depression": ["F3", "F4", "Fp1", "Fp2", "Cz", "Pz"],
    "epilepsy": ["F7", "F8", "T7", "T8", "Cz", "Pz"],
    "cognitive": ["Fz", "F3", "F4", "Cz", "Pz", "Oz"],
    "sleep": ["Fz", "Cz", "Pz", "O1", "O2", "T8"]
}

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50)
}

BAND_COLORS = {
    "delta": "#f44336",
    "theta": "#ff9800",
    "alpha": "#4caf50",
    "beta": "#2196f3",
    "gamma": "#9c27b0"
}

# ---------- Streamlit session state initialization ----------
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_mode = "alzheimer"
    st.session_state.channel_buffers = {}  # channel -> deque
    st.session_state.buf_maxlen = DISPLAY_BUFFER
    st.session_state.incoming_q = queue.Queue()
    st.session_state.ws_thread_started = False
    st.session_state.demo_mode = True  # default to demo; can toggle in sidebar
    st.session_state.auto_refresh = True
    st.session_state.refresh_rate = REFRESH_DEFAULT
    st.session_state.buffer_ready = False
    st.session_state.prediction_result = None
    st.session_state.last_buffer_status = {}
    st.session_state.ws_error = None
    st.session_state.demo_counter = 0


# ---------- Utility functions ----------
def get_mode_from_api():
    try:
        r = requests.get(f"{API_BASE}/api/mode", timeout=1.0)
        if r.ok:
            return r.json().get("mode", st.session_state.current_mode), r.json().get("channels", DETECTION_MODES.get(st.session_state.current_mode))
    except Exception:
        pass
    return st.session_state.current_mode, DETECTION_MODES[st.session_state.current_mode]


def set_mode_api(mode):
    try:
        r = requests.post(f"{API_BASE}/api/set_mode/{mode}", timeout=3.0)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_buffer_status():
    try:
        r = requests.get(f"{API_BASE}/api/buffer", timeout=1.0)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"ready": False, "samples": {}}


def call_predict_api():
    try:
        r = requests.post(f"{API_BASE}/api/predict", timeout=10.0)
        if r.ok:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as e:
        return {"error": str(e)}


def call_clear_api():
    try:
        requests.post(f"{API_BASE}/api/clear", timeout=2.0)
        return True
    except Exception:
        return False


# signal processing
def compute_psd_welch(samples):
    if len(samples) < 8:
        return np.array([]), np.array([])
    freqs, psd = signal.welch(np.asarray(samples[-FFT_SIZE:]), fs=SAMPLE_RATE, nperseg=min(FFT_SIZE, len(samples)))
    return freqs, psd


def compute_band_powers(samples):
    freqs, psd = compute_psd_welch(samples)
    out = {}
    if freqs.size == 0:
        for b in BANDS:
            out[b] = {"power": 0.0, "relative": 0.0}
        out["_total"] = 0.0
        return out
    total = float(np.trapz(psd, freqs))
    if total <= 0:
        total = 1e-12
    for b, (lo, hi) in BANDS.items():
        idx = (freqs >= lo) & (freqs <= hi)
        p = float(np.trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0
        out[b] = {"power": p, "relative": p / total}
    out["_total"] = total
    return out


def compute_time_features(samples):
    if len(samples) < 4:
        return {"mean": 0.0, "std": 0.0, "rms": 0.0, "median": 0.0, "skew": 0.0, "kurtosis": 0.0}
    a = np.asarray(samples[-FFT_SIZE:])
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "rms": float(np.sqrt(np.mean(a ** 2))),
        "median": float(np.median(a)),
        "skew": float(skew(a)),
        "kurtosis": float(kurtosis(a))
    }


# initialize buffers for a mode
def init_buffers_for_mode(mode):
    channels = DETECTION_MODES.get(mode, DETECTION_MODES["alzheimer"])
    for ch in channels:
        if ch not in st.session_state.channel_buffers:
            st.session_state.channel_buffers[ch] = deque(maxlen=st.session_state.buf_maxlen)
    # remove channels not in new mode
    for ch in list(st.session_state.channel_buffers.keys()):
        if ch not in channels:
            del st.session_state.channel_buffers[ch]
    st.session_state.current_mode = mode


# demo sample generator
def demo_sample(ch, t):
    # combine bands with small randomization + noise (scaled)
    amps = {"delta": 5.0, "theta": 3.0, "alpha": 4.0, "beta": 2.0, "gamma": 1.0}
    v = 0.0
    v += amps["delta"] * math.sin(2 * math.pi * 1.0 * t / SAMPLE_RATE + (hash(ch) % 10) * 0.01)
    v += amps["theta"] * math.sin(2 * math.pi * 6.0 * t / SAMPLE_RATE + 0.2)
    v += amps["alpha"] * math.sin(2 * math.pi * 10.0 * t / SAMPLE_RATE + 0.4)
    v += amps["beta"] * math.sin(2 * math.pi * 20.0 * t / SAMPLE_RATE + 0.6)
    v += amps["gamma"] * math.sin(2 * math.pi * 40.0 * t / SAMPLE_RATE + 0.8)
    v += np.random.normal(0, 0.6)
    return float(v)


# ---------- WebSocket background client (optional) ----------
def ws_on_open(wsapp):
    st.session_state.ws_error = None
    st.session_state.incoming_q.put({"type": "ws_open", "msg": "connected"})


def ws_on_message(wsapp, message):
    try:
        msg = json.loads(message)
    except Exception:
        return
    st.session_state.incoming_q.put(msg)


def ws_on_error(wsapp, error):
    st.session_state.ws_error = str(error)
    st.session_state.incoming_q.put({"type": "error", "msg": str(error)})


def ws_on_close(wsapp, close_status_code, close_msg):
    st.session_state.ws_error = f"closed: {close_status_code} {close_msg}"
    st.session_state.incoming_q.put({"type": "ws_close", "msg": st.session_state.ws_error})


def start_ws_thread(ws_url):
    if not WEBSOCKET_AVAILABLE:
        st.session_state.ws_error = "websocket-client not installed"
        return

    def run():
        while True:
            try:
                wsapp = websocket.WebSocketApp(ws_url,
                                               on_open=ws_on_open,
                                               on_message=ws_on_message,
                                               on_error=ws_on_error,
                                               on_close=ws_on_close)
                wsapp.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                st.session_state.ws_error = f"WS thread err: {e}"
                st.session_state.incoming_q.put({"type": "error", "msg": str(e)})
            time.sleep(2.0)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    st.session_state.ws_thread_started = True


# ---------- Drain queue and apply messages ----------
def drain_queue():
    q = st.session_state.incoming_q
    applied = 0
    while not q.empty():
        try:
            msg = q.get_nowait()
        except Exception:
            break
        applied += 1
        mtype = msg.get("type")
        if mtype == "sample":
            ch = msg.get("channel")
            if not ch and "index" in msg:
                idx = int(msg.get("index"))
                channels = list(DETECTION_MODES.get(st.session_state.current_mode))
                if 0 <= idx < len(channels):
                    ch = channels[idx]
            try:
                v = float(msg.get("value"))
            except Exception:
                continue
            if ch in st.session_state.channel_buffers:
                st.session_state.channel_buffers[ch].append(v)
        elif mtype == "batch":
            data = msg.get("data") or msg.get("values") or {}
            if isinstance(data, dict):
                for ch, vals in data.items():
                    if ch not in st.session_state.channel_buffers:
                        continue
                    if isinstance(vals, (list, tuple)):
                        for v in vals:
                            try:
                                st.session_state.channel_buffers[ch].append(float(v))
                            except Exception:
                                continue
                    else:
                        try:
                            st.session_state.channel_buffers[ch].append(float(vals))
                        except Exception:
                            continue
        elif mtype == "buffer":
            st.session_state.buffer_ready = bool(msg.get("ready"))
        elif mtype == "prediction":
            st.session_state.prediction_result = msg.get("result")
        elif mtype == "ws_open":
            st.session_state.ws_error = None
        elif mtype == "ws_close":
            st.session_state.ws_error = msg.get("msg")
        elif mtype == "error":
            st.session_state.ws_error = msg.get("msg")
    return applied


# ---------- UI building ----------
st.set_page_config(page_title="EEG Multi-Disorder Detection", page_icon="🧠", layout="wide")
st.markdown("# 🧠 EEG Real-Time Multi-Disorder Detection")
st.markdown("Classic stacked EEG view — live waveforms, band-powers, and model predictions")

# Sidebar
with st.sidebar:
    st.header("Controls")
    # Mode select
    mode_select = st.selectbox("Detection Mode", options=list(DETECTION_MODES.keys()),
                               index=list(DETECTION_MODES.keys()).index(st.session_state.current_mode),
                               format_func=lambda x: x.title())
    if st.button("Apply Mode"):
        res = set_mode_api(mode_select)
        init_buffers_for_mode(mode_select)
        st.success(f"Mode set to {mode_select}")
        # update remote and local buffers, then rerun
        st.rerun()

    st.divider()

    # Demo toggle
    demo_toggle = st.checkbox("Enable Demo Mode (no WS)", value=st.session_state.demo_mode)
    st.session_state.demo_mode = demo_toggle

    # WebSocket status + start thread if available and not demo
    if WEBSOCKET_AVAILABLE and not st.session_state.demo_mode:
        st.write("WebSocket client: available")
        if not st.session_state.ws_thread_started:
            start_ws_thread(WS_BASE)
            st.write("Started background WS client thread.")
        else:
            st.write("WS thread running.")
    else:
        if not WEBSOCKET_AVAILABLE:
            st.warning("websocket-client not installed — demo mode recommended")
        else:
            st.info("Demo mode active — no WS connection")

    st.divider()
    # Auto refresh controls
    st.session_state.auto_refresh = st.checkbox("Auto refresh UI", value=st.session_state.auto_refresh)
    st.session_state.refresh_rate = st.slider("Refresh interval (s)", min_value=0.2, max_value=2.0, step=0.1, value=float(st.session_state.refresh_rate))

    st.divider()
    # Actions
    if st.button("Predict Now"):
        st.session_state.prediction_result = call_predict_api()
        st.rerun()
    if st.button("Clear Buffers"):
        call_clear_api()
        for c in st.session_state.channel_buffers:
            st.session_state.channel_buffers[c].clear()
        st.session_state.prediction_result = None
        st.success("Cleared buffers")
        st.rerun()

    st.divider()
    st.markdown("Backend")
    st.markdown(f"- API: `{API_BASE}`")
    st.markdown(f"- WS: `{WS_BASE}`")
    if st.session_state.ws_error:
        st.error(f"WS: {st.session_state.ws_error}")

# initialize buffers for current mode
init_buffers_for_mode(st.session_state.current_mode)
channels = list(DETECTION_MODES[st.session_state.current_mode])

# If demo mode, add a few samples per refresh
if st.session_state.demo_mode:
    # add samples proportional to refresh_rate
    add_count = max(1, int(st.session_state.refresh_rate * SAMPLE_RATE / 4))
    for i in range(add_count):
        st.session_state.demo_counter += 1
        for ch in channels:
            st.session_state.channel_buffers[ch].append(demo_sample(ch, st.session_state.demo_counter))

# drain incoming queue from WS if not demo
if not st.session_state.demo_mode:
    drain_queue()

# get buffer status from server (non-blocking)
server_buf = get_buffer_status()
st.session_state.buffer_ready = server_buf.get("ready", st.session_state.buffer_ready)
st.session_state.last_buffer_status = server_buf.get("samples", st.session_state.last_buffer_status)

# ---------- Main layout: stacked vertical EEG strips ----------
st.markdown("## Live EEG — Classic Stacked View")
left_col, right_col = st.columns([3, 1])

with left_col:
    # Render each channel, stacked vertically
    for ch in channels:
        buf = list(st.session_state.channel_buffers.get(ch, []))
        # take last PLOT_WINDOW points for display
        if len(buf) >= 1:
            y = buf[-PLOT_WINDOW:]
            x = list(range(len(y)))
        else:
            x = list(range(PLOT_WINDOW))
            y = [0.0] * PLOT_WINDOW

        # auto-scale with padding
        if any(y):
            mn = min(y)
            mx = max(y)
            pad = (mx - mn) * 0.25 if (mx - mn) != 0 else 1.0
            ymin = mn - pad
            ymax = mx + pad
        else:
            ymin, ymax = -6, 6

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#00e676", width=1.6), hoverinfo="skip"))
        fig.update_layout(
            height=130,
            margin=dict(l=30, r=8, t=6, b=6),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(2,6,12,0.6)",
            xaxis=dict(visible=False),
            yaxis=dict(range=[ymin, ymax], showgrid=True, gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#9ab")),
            showlegend=False
        )
        # label overlay
        fig.update_layout(annotations=[dict(
            x=0.005, y=0.5, xref="paper", yref="paper",
            text=f"<b style='color:#bfefff'>{ch}</b>",
            showarrow=False, xanchor="left", font=dict(size=13)
        )])
        st.plotly_chart(fig, width="stretch")

        # band-power cards under waveform
        bp = compute_band_powers(buf)
        cols = st.columns(len(BANDS))
        for i, band in enumerate(BANDS.keys()):
            with cols[i]:
                info = bp[band]
                rel_pct = info["relative"] * 100
                st.markdown(
                    f"""
                    <div style="background:#071226;padding:8px;border-radius:8px;border:1px solid rgba(255,255,255,0.03)">
                        <div style="font-weight:700;color:{BAND_COLORS[band]};text-transform:uppercase">{band}</div>
                        <div style="font-size:12px;color:#9ab;margin-top:6px">{info['power']:.2e}</div>
                        <div style="height:8px;background:rgba(255,255,255,0.04);border-radius:4px;margin-top:6px;overflow:hidden">
                            <div style="width:{rel_pct:.2f}%;background:{BAND_COLORS[band]};height:100%"></div>
                        </div>
                        <div style="text-align:center;margin-top:6px;font-weight:700;color:#dcefff">{rel_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True
                )

        st.divider()

with right_col:
    st.markdown("### Buffer Status")
    if st.session_state.buffer_ready:
        st.success("✅ Buffer Ready (server has 1024 samples)")
    else:
        st.info("📡 Collecting samples...")
    if st.session_state.last_buffer_status:
        # display small summary table
        samp_df = pd.DataFrame(list(st.session_state.last_buffer_status.items()), columns=["Channel", "Samples"])
        st.table(samp_df)

    st.markdown("### Prediction")
    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        classes = res.get("classes", [])
        probs = res.get("class_probabilities", [])
        raw = res.get("prediction_raw")
        label = raw
        if classes and isinstance(raw, (int, float)):
            try:
                label = classes[int(raw)]
            except Exception:
                pass
        conf = res.get("confidence", max(probs) if probs else None)
        st.markdown(f"**Prediction:** <span style='color:#9ff'>{label}</span>", unsafe_allow_html=True)
        if conf is not None:
            st.markdown(f"**Confidence:** {conf*100:.1f}%")
            st.progress(min(1.0, float(conf)))
        if probs and classes and len(probs) == len(classes):
            dfp = pd.DataFrame({"class": classes, "prob": probs})
            dfp = dfp.sort_values("prob", ascending=False)
            st.table(dfp.style.format({"prob": "{:.3f}"}))
        st.markdown("**Interpretation**")
        st.write(res.get("explanation", "No additional explanation provided."))
    else:
        st.info("Run a prediction to display results here")

    st.markdown("### Controls")
    if st.button("Fetch Buffer Status"):
        st.rerun()
    st.markdown(f"- Demo mode: **{st.session_state.demo_mode}**")
    if WEBSOCKET_AVAILABLE:
        st.markdown("- WS client: available")
    else:
        st.markdown("- WS client: not installed (demo mode recommended)")

# ---------- Footer / auto-refresh logic ----------
if st.session_state.auto_refresh:
    time.sleep(float(st.session_state.refresh_rate))
    st.rerun()
