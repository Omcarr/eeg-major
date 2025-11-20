"""
Classic EEG stacked Streamlit dashboard (single file).
Features:
 - Polls Redis backend via HTTP for live data (no WebSocket needed)
 - ESP32-friendly: no broadcast overload
 - Classic vertical stacked EEG waveforms
 - Band-power cards and predictions
 - ONLY USES REAL BACKEND DATA
"""

from collections import deque
import os
import time
import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import signal
from scipy.stats import kurtosis, skew

API_BASE = os.environ.get("EEG_API_BASE", "http://localhost:8000")

GEMINI_API_KEY = "AIzaSyCqXa8wTLG8mKJXStEngTnte5aWVABKyz8"
GEMINI_MODEL = "gemini-1.5-flash"

def gemini_interpret(pred_label, conf, band_summary):
    """Generate a short, formatted interpretation using Gemini."""
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
PLOT_WINDOW = 512
REFRESH_DEFAULT = 0.5

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
    st.session_state.auto_refresh = True
    st.session_state.refresh_rate = REFRESH_DEFAULT
    st.session_state.buffer_ready = False
    st.session_state.prediction_result = None
    st.session_state.last_buffer_status = {}
    st.session_state.last_data_time = 0
    st.session_state.backend_status = "unknown"


# ---------- API functions ----------
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


def get_live_data(samples=512):
    """Get latest samples from Redis backend"""
    try:
        r = requests.get(f"{API_BASE}/api/live_data", params={"samples": samples}, timeout=2.0)
        if r.ok:
            return r.json()
    except Exception as e:
        st.session_state.backend_status = f"error: {e}"
    return None


def get_buffer_status():
    try:
        r = requests.get(f"{API_BASE}/api/buffer", timeout=1.0)
        if r.ok:
            st.session_state.backend_status = "connected"
            return r.json()
    except Exception as e:
        st.session_state.backend_status = f"error: {e}"
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


# ---------- UI building ----------
st.set_page_config(page_title="EEG Multi-Disorder Detection", page_icon="🧠", layout="wide")
st.markdown("# EEG Real-Time Multi-Disorder Detection")
# st.markdown("Classic stacked EEG view — live waveforms from Redis buffer")

# Sidebar
with st.sidebar:
    st.header("Controls")
    mode_select = st.selectbox(
        "Detection Mode",
        options=list(DETECTION_MODES.keys()),
        index=list(DETECTION_MODES.keys()).index(st.session_state.current_mode),
        format_func=lambda x: x.title()
    )
    if st.button("Apply Mode"):
        res = set_mode_api(mode_select)
        st.session_state.current_mode = mode_select
        st.success(f"Mode set to {mode_select}")
        st.rerun()

    st.divider()
    
    # Auto refresh controls
    st.session_state.auto_refresh = st.checkbox("Auto refresh UI", value=st.session_state.auto_refresh)
    st.session_state.refresh_rate = st.slider(
        "Refresh interval (s)",
        min_value=0.2,
        max_value=2.0,
        step=0.1,
        value=float(st.session_state.refresh_rate)
    )

    st.divider()
    
    # Actions
    if st.button("Predict Now"):
        st.session_state.prediction_result = call_predict_api()
        st.rerun()
        
    if st.button("Clear Buffers"):
        call_clear_api()
        st.session_state.prediction_result = None
        st.success("Cleared buffers")
        st.rerun()

channels = list(DETECTION_MODES[st.session_state.current_mode])

live_data_response = get_live_data(samples=PLOT_WINDOW)
channel_data = {}
has_live_data = False

if live_data_response:
    st.session_state.last_data_time = time.time()
    channel_data = live_data_response.get("data", {})
    has_live_data = bool(channel_data)

server_buf = get_buffer_status()
st.session_state.buffer_ready = server_buf.get("ready", False)
st.session_state.last_buffer_status = server_buf.get("samples", {})

st.markdown("## Live EEG data")

# Show warning if no data
if not has_live_data:
    st.warning("⚠️ No data from backend. Make sure ESP32 is connected and sending data.")

left_col, right_col = st.columns([3, 1])

with left_col:
    # Render each channel stacked vertically
    for ch in channels:
        # Get data for this channel (Redis returns newest first)
        y = channel_data.get(ch, [])
        has_data = len(y) > 0
        
        # Redis returns newest first, so reverse for chronological display
        if has_data:
            y = list(reversed(y[:PLOT_WINDOW]))
            x = list(range(len(y)))
        else:
            x = list(range(PLOT_WINDOW))
            y = [0.0] * PLOT_WINDOW

        # Auto-scale with padding
        if has_data and any(y):
            mn = min(y)
            mx = max(y)
            pad = (mx - mn) * 0.25 if (mx - mn) != 0 else 1.0
            ymin = mn - pad
            ymax = mx + pad
        else:
            ymin, ymax = -10, 10

        fig = go.Figure()
        
        # Waveform
        line_color = "#00e676" if has_data else "#404040"
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=line_color, width=1.6),
            hoverinfo="skip"
        ))
        
        fig.update_layout(
            height=130,
            margin=dict(l=30, r=8, t=6, b=6),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(2,6,12,0.6)",
            xaxis=dict(visible=False),
            yaxis=dict(
                range=[ymin, ymax],
                showgrid=True,
                gridcolor="rgba(255,255,255,0.03)",
                tickfont=dict(color="#9ab")
            ),
            showlegend=False
        )
        
        # Channel label with status
        status_dot = "🟢" if has_data else "⚪"
        fig.update_layout(annotations=[dict(
            x=0.005, y=0.5,
            xref="paper", yref="paper",
            text=f"<b style='color:#bfefff'>{status_dot} {ch}</b>",
            showarrow=False,
            xanchor="left",
            font=dict(size=13)
        )])
        
        # "Waiting" message if no data
        if not has_data:
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="Waiting for data from ESP32...",
                showarrow=False,
                font=dict(size=11, color="#666"),
                xanchor="center"
            )
        
        st.plotly_chart(fig, use_container_width=True)

        # Band-power cards (only if we have data)
        if has_data:
            bp = compute_band_powers(y)
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
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.caption("⏳ Band powers will appear once data is received")

        st.divider()

with right_col:
    st.markdown("### Buffer Status")
    if st.session_state.buffer_ready:
        st.success("✅ Buffer Ready")
    else:
        st.info("📡 Collecting samples...")
    
    if st.session_state.last_buffer_status:
        samp_df = pd.DataFrame(
            list(st.session_state.last_buffer_status.items()),
            columns=["Channel", "Samples"]
        )
        st.table(samp_df)
    else:
        st.caption("No buffer data yet")

    st.markdown("### Prediction")
    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        
        if "error" in res:
            st.error(f"❌ {res['error']}")
        else:
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
            st.write(res.get("explanation", "No explanation provided."))
    else:
        st.info("Run prediction to see results")

    # st.markdown("### Data Flow")
    # st.caption("ESP32 → WebSocket → Redis")
    # st.caption("Dashboard ← HTTP Poll ← Redis")
    # st.caption("🚀 No broadcast overhead")

# Auto-refresh
if st.session_state.auto_refresh:
    time.sleep(float(st.session_state.refresh_rate))
    st.rerun()