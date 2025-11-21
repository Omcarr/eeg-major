# ui.py
"""
Complete Streamlit UI (single-file) for EEG Real-Time Multi-Disorder Detection
+ Alpha-based Emotional State module (Option A: keep disorder prediction and add emotion block)
Includes an example embedded image (from uploaded file path).
"""

from collections import deque
import os
import time
import requests
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import signal

# ============================================================
# CONFIG
# ============================================================

API_BASE = os.environ.get("EEG_API_BASE", "http://localhost:8000")

# ---- Gemini (prefer env var; fallback to placeholder) ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCqXa8wTLG8mKJXStEngTnte5aWVABKyz8")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

# Path to uploaded image (provided by developer/instruction)
# EXAMPLE_IMAGE_PATH = "/mnt/data/e51ad7c5-dc98-43fd-8ad9-35025a7ad699.png"

SAMPLE_RATE = 128
FFT_SIZE = 256
PLOT_WINDOW = 512
REFRESH_DEFAULT = 2.0

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
    "gamma": (30, 50),
}

BAND_COLORS = {
    "delta": "#f44336",
    "theta": "#ff9800",
    "alpha": "#4caf50",
    "beta": "#2196f3",
    "gamma": "#9c27b0",
}

EMOTIONS = ["Relaxed", "Focused", "Stressed", "Drowsy"]

# ============================================================
# HELPERS: SIGNAL PROCESSING & BANDS
# ============================================================

def compute_psd_welch(samples):
    if len(samples) < 8:
        return np.array([]), np.array([])
    freqs, psd = signal.welch(
        np.asarray(samples[-FFT_SIZE:]),
        fs=SAMPLE_RATE,
        nperseg=min(FFT_SIZE, len(samples)),
    )
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
        p = float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0
        out[b] = {"power": p, "relative": p / total}

    out["_total"] = total
    return out

def summarize_bands_for_interpretation(channel_data):
    """Compute mean band powers and relative values across channels."""
    band_power = {b: [] for b in BANDS.keys()}
    band_rel = {b: [] for b in BANDS.keys()}

    for ch, samples in channel_data.items():
        bp = compute_band_powers(samples)
        for b in BANDS.keys():
            band_power[b].append(bp[b]["power"])
            band_rel[b].append(bp[b]["relative"])

    summary = {}
    for b in BANDS.keys():
        summary[b] = {
            "mean_power": float(np.mean(band_power[b])) if band_power[b] else 0.0,
            "mean_relative": float(np.mean(band_rel[b])) if band_rel[b] else 0.0,
        }
    return summary

# ============================================================
# EMOTIONAL STATE MODEL (ALPHA-DRIVEN)
# ============================================================

def compute_emotion_distribution(alpha_rel):
    """
    Heuristic probability distribution over emotional states based on alpha relative power.
    alpha_rel: float in [0,1] — mean relative alpha power across relevant channels.
    Returns dict of emotion -> probability (sums to 1).
    """
    # Heuristic: alpha increase -> more relaxed/drowsy; alpha decrease -> more focused/stressed
    relaxed = alpha_rel * 0.55 + 0.05        # scales with alpha
    drowsy = alpha_rel * 0.30 + 0.02         # scales with alpha
    # complementary mass distributed to focused/stressed
    focused = (1 - alpha_rel) * 0.45 + 0.02
    stressed = (1 - alpha_rel) * 0.25 + 0.02

    raw = np.array([relaxed, focused, stressed, drowsy])
    probs = raw / raw.sum()
    return {emo: float(probs[i]) for i, emo in enumerate(EMOTIONS)}

def gemini_emotion_interpret(alpha_level, emotion_probs):
    """
    Short 2-3 line emotional interpretation via Gemini (if available).
    Falls back to a concise local string on error.
    """
    # prompt = (
    #     "You are an EEG emotional-state assistant.\n"
    #     f"Alpha relative intensity: {alpha_level:.3f}\n"
    #     f"Emotion distribution: {json.dumps(emotion_probs)}\n\n"
    #     "Write a VERY SHORT 2–3 line clinical-style interpretation of the user's current "
    #     "emotional state based primarily on alpha activity. Keep it concise and actionable.\n"
    #     "Format:\n- Line1\n- Line2\n"
    # )

    # try:
    #     url = (
    #         f"https://generativelanguage.googleapis.com/v1beta/models/"
    #         f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    #     )
    #     headers = {"Content-Type": "application/json"}
    #     payload = {"contents": [{"parts": [{"text": prompt}]}]}
    #     res = requests.post(url, headers=headers, json=payload, timeout=6)
    #     data = res.json()
    #     # defensive access
    #     text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    #     if not text:
    #         raise ValueError("Empty response from Gemini")
    #     return text.strip()
    # except Exception as e:
    #     # Provide a compact fallback interpretation
    #     top = max(emotion_probs.items(), key=lambda x: x[1])
    #     fallback = [
    #         f"- Dominant emotional state: {top[0]} ({top[1]*100:.1f}%).",
    #         f"- Alpha level suggests {('higher' if alpha_level>0.15 else 'lower')} cortical alpha activation."
    #     ]
    #     return "\n".join(fallback) + f"\n- Gemini error: {e}"
    top = max(emotion_probs.items(), key=lambda x: x[1])
    fallback = [
        f"- Dominant emotional state: {top[0]} ({top[1]*100:.1f}%).",
        f"- Alpha level suggests {('higher' if alpha_level>0.15 else 'lower')} cortical alpha activation."
    ]
    return "\n".join(fallback)

# ============================================================
# API FUNCTIONS
# ============================================================

def set_mode_api(mode):
    try:
        r = requests.post(f"{API_BASE}/api/set_mode/{mode}", timeout=3)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_live_data(samples=512):
    try:
        r = requests.get(f"{API_BASE}/api/live_data", params={"samples": samples}, timeout=3)
        if r.ok:
            return r.json()
    except:
        pass
    return None

def get_buffer_status():
    try:
        r = requests.get(f"{API_BASE}/api/buffer", timeout=1)
        if r.ok:
            return r.json()
    except:
        pass
    return {"ready": False, "samples": {}}

def call_predict_api():
    try:
        r = requests.post(f"{API_BASE}/api/predict", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def call_clear_api():
    try:
        requests.post(f"{API_BASE}/api/clear", timeout=3)
        return True
    except:
        return False

# ============================================================
# SESSION STATE
# ============================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_mode = "alzheimer"
    st.session_state.auto_refresh = True
    st.session_state.refresh_rate = REFRESH_DEFAULT
    st.session_state.prediction_result = None

# ============================================================
# UI SETUP
# ============================================================

st.set_page_config(page_title="EEG Multi-Disorder Detection + Emotion", page_icon="🧠", layout="wide")
st.markdown("# EEG Real-Time Multi-Disorder Detection")

with st.sidebar:
    st.header("Controls")
    mode_select = st.selectbox(
        "Detection Mode",
        options=list(DETECTION_MODES.keys()),
        index=list(DETECTION_MODES.keys()).index(st.session_state.current_mode),
    )
    if st.button("Apply Mode"):
        set_mode_api(mode_select)
        st.session_state.current_mode = mode_select
        st.success(f"Mode set to {mode_select}")
        st.rerun()

    st.session_state.auto_refresh = st.checkbox("Auto-refresh", st.session_state.auto_refresh)
    st.session_state.refresh_rate = st.slider("Refresh interval", 0.2, 5.0, st.session_state.refresh_rate, 0.1)

    if st.button("Predict"):
        st.session_state.prediction_result = call_predict_api()
        st.rerun()

    if st.button("Clear buffers"):
        call_clear_api()
        st.session_state.prediction_result = None
        st.rerun()


# ============================================================
# Live Data Acquisition
# ============================================================

channels = DETECTION_MODES[st.session_state.current_mode]
live_data = get_live_data(samples=PLOT_WINDOW)
channel_data = live_data.get("data", {}) if live_data else {}
ready = get_buffer_status().get("ready", False)

st.markdown("## Live EEG Signals")

if not channel_data:
    st.warning("Waiting for live EEG data from backend... (UI will show zeros until data arrives)")

left_col, right_col = st.columns([3, 1])

# ============================================================
# LEFT COLUMN – EEG WAVEFORMS + BAND CARDS
# ============================================================

with left_col:
    # Plot each configured channel as a stacked waveform
    for ch in channels:
        y = channel_data.get(ch, [])
        has_data = len(y) > 0

        if has_data:
            # ensure list length
            y = list(reversed(y[:PLOT_WINDOW]))
            x = list(range(len(y)))
        else:
            x = list(range(PLOT_WINDOW))
            y = [0.0] * PLOT_WINDOW

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#00e676", width=1.4)))

        mn, mx = min(y), max(y)
        pad = (mx - mn) * 0.25 if (mx - mn) else 1
        fig.update_layout(
            height=130,
            margin=dict(l=25, r=10, t=5, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(2,6,12,0.6)",
            xaxis=dict(visible=False),
            yaxis=dict(range=[mn - pad, mx + pad], showgrid=True, gridcolor="rgba(255,255,255,0.03)"),
        )

        fig.update_layout(annotations=[dict(
            x=0.005, y=0.5,
            xref="paper", yref="paper",
            text=f"🧩 <b>{ch}</b>",
            font=dict(color="#bfefff"),
            showarrow=False,
            xanchor="left"
        )])

        st.plotly_chart(fig, use_container_width=True)

        # Band-power cards
        if has_data:
            bp = compute_band_powers(y)
            cols = st.columns(len(BANDS))
            for i, b in enumerate(BANDS.keys()):
                with cols[i]:
                    p = bp[b]["power"]
                    rel = bp[b]["relative"] * 100
                    st.markdown(
                        f"""
                        <div style="background:#081420;padding:8px;border-radius:8px">
                            <div style="color:{BAND_COLORS[b]};font-weight:700;text-transform:uppercase">{b}</div>
                            <div style="font-size:12px;color:#9ab">{p:.2e}</div>
                            <div style="height:8px;background:#333;border-radius:4px;margin-top:6px">
                                <div style="width:{rel:.1f}%;height:100%;background:{BAND_COLORS[b]};border-radius:4px"></div>
                            </div>
                            <div style="text-align:center;margin-top:4px;color:#dcefff;font-weight:600">
                                {rel:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        st.divider()

# ============================================================
# RIGHT COLUMN – PREDICTION + EMOTIONAL STATE
# ============================================================

with right_col:
    st.markdown("### Prediction")

    res = st.session_state.prediction_result

    if not res:
        st.info("Run prediction to see results.")
    elif "error" in res:
        st.error(f"❌ {res['error']}")
    else:
        raw = res.get("prediction_raw")
        classes = res.get("classes", [])
        probs = res.get("class_probabilities", [])

        # Convert raw index → label if possible
        if classes and isinstance(raw, int) and 0 <= raw < len(classes):
            pred_label = classes[raw]
        else:
            pred_label = str(raw) if raw is not None else "Unknown"

        confidence = res.get("confidence", None)
        if confidence is None:
            # fallback: use max class probability if present
            confidence = float(max(probs)) if probs else 0.0

        st.markdown(f"""
        **Prediction:** <span style='color:#9ff'>{pred_label}</span><br>
        **Confidence:** {confidence*100:.1f}%
        """, unsafe_allow_html=True)
        st.progress(float(confidence))

        # Band summary for interpretation
        band_summary = summarize_bands_for_interpretation(channel_data)

        # Long explanation (local)
        # Determine dominant band
        dom = sorted(band_summary.items(), key=lambda x: x[1]["mean_relative"], reverse=True)
        top_band = dom[0][0] if dom else "unknown"

        descriptive_lines = []
        if pred_label=="1":
            descriptive_lines.append(f"Healthy (confidence {confidence*100:.1f}%).")
        else:
            descriptive_lines.append(f"Detected disorder: **{pred_label}** (confidence {confidence*100:.1f}%).")

        descriptive_lines.append(f"Dominant EEG band: **{top_band.upper()}**.")

        mode = st.session_state.current_mode
        if mode == "alzheimer":
            descriptive_lines.append("Alzheimer-mode focuses on posterior rhythms.")
            if top_band == "theta":
                descriptive_lines.append("Elevated theta may indicate cognitive slowing or early decline.")
            elif top_band == "delta":
                descriptive_lines.append("Delta prominence can signal cortical dysfunction.")
            elif top_band == "alpha":
                descriptive_lines.append("Healthy alpha levels suggest preserved resting-state networks.")
        elif mode == "parkinson":
            descriptive_lines.append("Parkinson-mode examines motor circuit beta abnormalities.")
            if top_band == "beta":
                descriptive_lines.append("Beta dominance can be associated with motor rigidity patterns.")
        elif mode == "depression":
            descriptive_lines.append("Depression-mode analyzes frontal alpha asymmetry.")
            if top_band == "alpha":
                descriptive_lines.append("High alpha may reflect frontal hypoactivation.")
        else:
            descriptive_lines.append("General EEG interpretation applied.")

        st.markdown("### Interpretation")
        for l in descriptive_lines:
            st.write(f"- {l}")

        # Gemini short summary (clinical)
        # try:
        #     gem_prompt = (
        #         f"You are an EEG analysis assistant.\nPrediction: {pred_label}\nConfidence: {confidence*100:.1f}%\n"
        #         f"Band summary (means): {json.dumps(band_summary)}\n"
        #         "Write a VERY SHORT clinical-style interpretation (2-3 lines, no filler).\n"
        #         "Format:\n- Insight 1\n- Insight 2\n"
        #     )
        #     gem_url = (
        #         f"https://generativelanguage.googleapis.com/v1beta/models/"
        #         f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        #     )
        #     gem_headers = {"Content-Type": "application/json"}
        #     gem_payload = {"contents": [{"parts": [{"text": gem_prompt}]}]}
        #     gem_res = requests.post(gem_url, headers=gem_headers, json=gem_payload, timeout=6)
        #     gem_data = gem_res.json()
        #     gem_text = gem_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        #     if not gem_text:
        #         raise ValueError("Empty AI response")
        #     gem_summary = gem_text.strip()
        # except Exception as e:
        #     gem_summary = f"- AI interpretation unavailable\n- Error: {e}"

        # st.markdown("### AI Summary")
        # st.info(gem_summary)

    # -----------------------------
    # Emotional-state (ALPHA-based)
    # -----------------------------
    st.markdown("### Emotional State (Alpha-based)")

    # compute band summary even if no prediction ran
    band_summary = summarize_bands_for_interpretation(channel_data)

    alpha_mean = float(band_summary.get("alpha", {}).get("mean_relative", 0.0))
    emotion_probs = compute_emotion_distribution(alpha_mean)

    # Display percentages
    for emo, p in emotion_probs.items():
        st.write(f"**{emo}:** {p*100:.1f}%")

    # # Show a small bar-like progress visualization
    # for emo, p in emotion_probs.items():
    #     st.progress(p)

    # Gemini emotion micro-interpretation
    gem_emotion_text = gemini_emotion_interpret(alpha_mean, emotion_probs)
    st.markdown("### AI Emotional Interpretation")
    st.info(gem_emotion_text)

# ============================================================
# AUTO REFRESH
# ============================================================

if st.session_state.auto_refresh:
    # simple sleep + rerun cycle for live feel
    time.sleep(st.session_state.refresh_rate)
    st.rerun()
