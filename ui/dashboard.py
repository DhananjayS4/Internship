"""
EASD Premium Dashboard — Real-Time Anxiety & Sleep Monitoring
Run from project root:
    streamlit run ui/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from joblib import load
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from processing.filtering import bandpass_emg, lowpass_accel
from processing.features import extract_emg_features, extract_accel_features

# ─── SENSOR IMPORTS (graceful fallback for dev machine) ───────────────────────
SENSORS_AVAILABLE = False
try:
    from drivers.emg_adc import EMGADC
    from drivers.imu_mpu6050 import MPU6050Driver
    SENSORS_AVAILABLE = True
except Exception:
    pass

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_V2    = os.path.join(BASE_DIR, "ml", "models", "anxiety_model_v2.joblib")
MODEL_V1    = os.path.join(BASE_DIR, "ml", "models", "anxiety_model.joblib")
METRICS_F   = os.path.join(BASE_DIR, "data", "model_metrics.json")
EXPORT_PATH = os.path.join(BASE_DIR, "data", "live_predictions.csv")

EMG_FS       = 200
ACC_FS       = 50
WINDOW_SEC   = 30
HISTORY_LEN  = 120
EMG_CHUNK    = 40
ACC_CHUNK    = 10

DARK_BG      = "#0a0e1a"
CARD_BG      = "rgba(255,255,255,0.05)"
ACCENT       = "#7c3aed"
STRESS_RED   = "#ef4444"
CALM_GREEN   = "#22c55e"
WARN_YELLOW  = "#f59e0b"

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EASD · Anxiety & Sleep Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DARK_BG};
    color: #e2e8f0;
}}

/* Main background */
.stApp {{
    background: linear-gradient(135deg, {DARK_BG} 0%, #0f1629 50%, #0d1117 100%);
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
    border-right: 1px solid rgba(124,58,237,0.3);
}}

/* Glass cards */
.glass-card {{
    background: {CARD_BG};
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}}

/* Metric big number */
.metric-value {{
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1;
}}
.metric-label {{
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    margin-bottom: 6px;
}}
.metric-sub {{
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
}}

/* Stress ring pulse animation */
@keyframes pulse-ring {{
    0%   {{ transform: scale(1);   opacity: 0.8; }}
    50%  {{ transform: scale(1.08); opacity: 0.4; }}
    100% {{ transform: scale(1);   opacity: 0.8; }}
}}
.pulse-ring {{
    display: inline-block;
    width: 14px; height: 14px;
    border-radius: 50%;
    animation: pulse-ring 1.5s ease-in-out infinite;
    margin-right: 8px;
    vertical-align: middle;
}}
.pulse-green  {{ background: {CALM_GREEN}; box-shadow: 0 0 10px {CALM_GREEN}; }}
.pulse-yellow {{ background: {WARN_YELLOW}; box-shadow: 0 0 10px {WARN_YELLOW}; }}
.pulse-red    {{ background: {STRESS_RED};  box-shadow: 0 0 10px {STRESS_RED};  }}

/* Status badge */
.status-badge {{
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
.badge-green  {{ background: rgba(34,197,94,0.15);  color: {CALM_GREEN}; border: 1px solid {CALM_GREEN}; }}
.badge-yellow {{ background: rgba(245,158,11,0.15); color: {WARN_YELLOW}; border: 1px solid {WARN_YELLOW}; }}
.badge-red    {{ background: rgba(239,68,68,0.15);  color: {STRESS_RED};  border: 1px solid {STRESS_RED};  }}

/* Section headers */
.section-header {{
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #7c3aed;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(124,58,237,0.5), transparent);
}}

/* Plotly chart background override */
.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

/* Streamlit metric */
[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 16px;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(124,58,237,0.4);
}}

/* Hide default Streamlit branding */
#MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─── AUTO REFRESH ─────────────────────────────────────────────────────────────
st_autorefresh(interval=1000, key="easd_refresh")

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = MODEL_V2 if os.path.exists(MODEL_V2) else MODEL_V1
    return load(path), path

model, model_path = load_model()
model_version = "v2 · WESAD" if "v2" in model_path else "v1 · Synthetic"

# Load metrics if available
model_metrics = {}
if os.path.exists(METRICS_F):
    with open(METRICS_F) as f:
        model_metrics = json.load(f)

# ─── SESSION STATE INIT ────────────────────────────────────────────────────────
defaults = {
    "emg_buffer":      [],
    "acc_buffer":      [],
    "stress_history":  deque(maxlen=HISTORY_LEN),
    "bpm_history":     deque(maxlen=HISTORY_LEN),
    "session_log":     [],
    "recording":       True,
    "session_start":   datetime.now().strftime("%H:%M:%S"),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── INIT SENSORS ─────────────────────────────────────────────────────────────
if SENSORS_AVAILABLE:
    if "emg_sensor" not in st.session_state:
        st.session_state.emg_sensor = EMGADC()
    if "mpu_sensor" not in st.session_state:
        st.session_state.mpu_sensor = MPU6050Driver()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 24px">
        <div style="font-size:2.2rem;">🧠</div>
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">EASD Monitor</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:4px;">
            Embedded Anxiety & Sleep Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    session_name = st.text_input("📝 Session Name", value="Morning Session")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶ Start" if not st.session_state.recording else "⏸ Pause"):
            st.session_state.recording = not st.session_state.recording
    with col_s2:
        if st.button("🔄 Reset"):
            st.session_state.emg_buffer = []
            st.session_state.acc_buffer = []
            st.session_state.stress_history.clear()
            st.session_state.bpm_history.clear()
            st.session_state.session_log = []
            st.session_state.session_start = datetime.now().strftime("%H:%M:%S")

    st.markdown("---")

    # Model info
    st.markdown("**🤖 Model Info**")
    acc_pct = f"{model_metrics.get('best_accuracy', 0)*100:.1f}%" if model_metrics else "—"
    f1_pct  = f"{model_metrics.get('best_f1', 0)*100:.1f}%"      if model_metrics else "—"
    st.markdown(f"""
    <div class="glass-card" style="padding:12px 16px">
        <div style="font-size:0.78rem; color:#94a3b8;">Version</div>
        <div style="font-size:0.9rem; font-weight:600; color:#7c3aed;">{model_version}</div>
        <div style="margin-top:8px; display:flex; justify-content:space-between;">
            <div>
                <div style="font-size:0.72rem; color:#64748b;">Accuracy</div>
                <div style="font-size:1rem; font-weight:600; color:#22c55e;">{acc_pct}</div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#64748b;">F1 Score</div>
                <div style="font-size:1rem; font-weight:600; color:#22c55e;">{f1_pct}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model_metrics:
        best_model = model_metrics.get("model_name", "—")
        data_src   = model_metrics.get("data_source", "—").upper()
        n_samp     = model_metrics.get("n_samples", "—")
        st.caption(f"Algorithm: **{best_model}**\nData: **{data_src}** ({n_samp} samples)")

    st.markdown("---")
    if st.button("💾 Export Session CSV"):
        if st.session_state.session_log:
            df_export = pd.DataFrame(st.session_state.session_log)
            df_export.to_csv(EXPORT_PATH, index=False)
            st.success(f"Saved to `{EXPORT_PATH}`")
        else:
            st.warning("No data to export yet.")

    st.markdown("---")
    st.caption(f"🟢 Sensors: {'Connected' if SENSORS_AVAILABLE else 'Demo mode'}")
    st.caption(f"⏱ Session start: {st.session_state.session_start}")

# ─── TITLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 20px">
    <h1 style="font-size:1.8rem; font-weight:700; margin:0; color:#e2e8f0;">
        🧠 Real-Time Anxiety & Sleep Monitor
    </h1>
    <p style="color:#64748b; margin:4px 0 0; font-size:0.88rem;">
        Continuous EMG · Accelerometry · HRV · Sleep Stage Classification
    </p>
</div>
""", unsafe_allow_html=True)

# ─── DATA COLLECTION ──────────────────────────────────────────────────────────
if SENSORS_AVAILABLE and st.session_state.recording:
    emg_s = st.session_state.emg_sensor
    mpu_s = st.session_state.mpu_sensor
    for _ in range(EMG_CHUNK):
        st.session_state.emg_buffer.append(emg_s.read_voltage())
    for _ in range(ACC_CHUNK):
        acc = mpu_s.read_accel()
        st.session_state.acc_buffer.append([acc["x"], acc["y"], acc["z"]])
elif not SENSORS_AVAILABLE:
    # Demo: generate realistic synthetic signals
    t = len(st.session_state.emg_buffer)
    for i in range(EMG_CHUNK):
        noise = np.random.normal(0, 5e-5)
        st.session_state.emg_buffer.append(noise + 3e-5 * np.sin(2 * np.pi * 50 * (t + i) / EMG_FS))
    for i in range(ACC_CHUNK):
        st.session_state.acc_buffer.append([
            9.81 + np.random.normal(0, 0.05),
            np.random.normal(0, 0.03),
            np.random.normal(0, 0.03),
        ])

# Trim buffers
st.session_state.emg_buffer = st.session_state.emg_buffer[-(EMG_FS * WINDOW_SEC):]
st.session_state.acc_buffer = st.session_state.acc_buffer[-(ACC_FS * WINDOW_SEC):]

# Check buffer readiness
buf_emg_pct = len(st.session_state.emg_buffer) / (EMG_FS * WINDOW_SEC)
buf_acc_pct = len(st.session_state.acc_buffer) / (ACC_FS * WINDOW_SEC)

if buf_emg_pct < 1.0 or buf_acc_pct < 1.0:
    warmup_pct = min(buf_emg_pct, buf_acc_pct)
    st.markdown(f"""
    <div class="glass-card" style="text-align:center; padding:32px;">
        <div style="font-size:2rem;">⏳</div>
        <div style="font-size:1.1rem; font-weight:600; margin:8px 0;">Warming up sensors...</div>
        <div style="font-size:0.85rem; color:#64748b;">Collecting {WINDOW_SEC}s of baseline data</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(warmup_pct, text=f"{warmup_pct*100:.0f}% ready")
    st.stop()

# ─── SIGNAL PROCESSING ────────────────────────────────────────────────────────
emg_arr  = np.array(st.session_state.emg_buffer)
acc_arr  = np.array(st.session_state.acc_buffer)

emg_f    = bandpass_emg(emg_arr, EMG_FS)
acc_mag  = np.linalg.norm(acc_arr, axis=1)
acc_f    = lowpass_accel(acc_mag, ACC_FS)

f_emg    = extract_emg_features(emg_f)
f_acc    = extract_accel_features(acc_arr)

# Determine feature count from model
feature_row = [
    f_emg["emg_rms"], f_emg["emg_var"], f_emg["emg_mean"],
    f_acc["acc_mean"], f_acc["acc_std"],  f_acc["acc_max"],
]
# If model expects 10 features (WESAD), pad with HRV/EDA zeros
try:
    n_feats = model.n_features_in_
except AttributeError:
    try:
        n_feats = model.steps[-1][1].n_features_in_
    except Exception:
        n_feats = 6

if n_feats == 10:
    # Use neutral/mid-range HRV values instead of 0.0 to avoid stressing the model
    # Typical resting values: rmssd~0.04s, sdnn~0.05s, pnn50~0.3, eda~2.0uS
    feature_row += [0.04, 0.05, 0.30, 2.0]  # hrv_rmssd, hrv_sdnn, hrv_pnn50, eda_mean

features = [feature_row]

# ─── PREDICTION ───────────────────────────────────────────────────────────────
try:
    stress_prob = float(model.predict_proba(features)[0][1])
except Exception:
    stress_prob = float(model.predict(features)[0]) * 0.9

st.session_state.stress_history.append(stress_prob)

# BPM estimate from acc (motion proxy)
# Use a gentler scaling: typical resting acc_f.std ~ 0.01-0.05 m/s²
# Scale so that std=0 -> 60 BPM, std=0.5 -> ~110 BPM (physiologically reasonable)
acc_std_val = float(acc_f.std())
bpm_est = max(50, min(110, int(60 + acc_std_val * 100)))
st.session_state.bpm_history.append(bpm_est)

# Sleep stage
def classify_sleep(emg_rms, acc_std, stress_prob):
    if acc_std > 0.4:
        return "AWAKE", "🌅"
    if emg_rms < 2e-5 and acc_std < 0.05:
        return "DEEP SLEEP (N3)", "💤"
    if emg_rms < 3e-5 and stress_prob < 0.3:
        return "LIGHT SLEEP (N2)", "🌙"
    return "REM SLEEP", "💭"

sleep_stage, sleep_icon = classify_sleep(f_emg["emg_rms"], f_acc["acc_std"], stress_prob)

# Log
if st.session_state.recording:
    st.session_state.session_log.append({
        "timestamp": datetime.now().isoformat(),
        "stress_prob": round(stress_prob, 4),
        "bpm": bpm_est,
        "sleep_stage": sleep_stage,
        "emg_rms": f_emg["emg_rms"],
        "acc_std": f_acc["acc_std"],
    })

# ─── STATUS COLOURS ───────────────────────────────────────────────────────────
if stress_prob < 0.3:
    status_label = "LOW STRESS"
    pulse_cls    = "pulse-green"
    badge_cls    = "badge-green"
    stress_color = CALM_GREEN
elif stress_prob < 0.65:
    status_label = "MODERATE STRESS"
    pulse_cls    = "pulse-yellow"
    badge_cls    = "badge-yellow"
    stress_color = WARN_YELLOW
else:
    status_label = "HIGH STRESS"
    pulse_cls    = "pulse-red"
    badge_cls    = "badge-red"
    stress_color = STRESS_RED

# ─── TOP METRIC CARDS ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Live Status</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Stress Level</div>
        <div style="margin: 8px 0">
            <span class="pulse-ring {pulse_cls}"></span>
            <span class="status-badge {badge_cls}">{status_label}</span>
        </div>
        <div class="metric-value" style="color:{stress_color}; font-size:2.4rem;">
            {stress_prob*100:.1f}<span style="font-size:1rem; color:#64748b;">%</span>
        </div>
        <div class="metric-sub">Stress probability</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Heart Rate</div>
        <div class="metric-value" style="color:#f472b6;">{bpm_est}
            <span style="font-size:1rem; color:#64748b;">BPM</span>
        </div>
        <div class="metric-sub">Estimated from motion</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Sleep Stage</div>
        <div style="font-size:1.8rem; margin:6px 0">{sleep_icon}</div>
        <div style="font-size:1rem; font-weight:600; color:#e2e8f0;">{sleep_stage}</div>
        <div class="metric-sub">EMG + Motion based</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    avg_stress = np.mean(st.session_state.stress_history) if st.session_state.stress_history else 0
    peak_stress = np.max(st.session_state.stress_history) if st.session_state.stress_history else 0
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Session Stats</div>
        <div style="margin-top:8px">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px">
                <span style="color:#94a3b8; font-size:0.8rem;">Avg Stress</span>
                <span style="font-weight:600;">{avg_stress*100:.1f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px">
                <span style="color:#94a3b8; font-size:0.8rem;">Peak Stress</span>
                <span style="font-weight:600; color:{STRESS_RED};">{peak_stress*100:.1f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between">
                <span style="color:#94a3b8; font-size:0.8rem;">Samples</span>
                <span style="font-weight:600;">{len(st.session_state.session_log)}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── STRESS GAUGE ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Stress Gauge & Sleep Distribution</div>', unsafe_allow_html=True)
g1, g2 = st.columns([1, 1])

with g1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=stress_prob * 100,
        delta={"reference": 50, "valueformat": ".1f", "suffix": "%"},
        title={"text": "Stress Probability", "font": {"color": "#94a3b8", "size": 13}},
        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar":  {"color": stress_color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 30],   "color": "rgba(34,197,94,0.12)"},
                {"range": [30, 65],  "color": "rgba(245,158,11,0.12)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": stress_color, "width": 3},
                          "thickness": 0.85, "value": stress_prob * 100},
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with g2:
    # Sleep stage donut (from session log)
    if st.session_state.session_log:
        stages = [r["sleep_stage"] for r in st.session_state.session_log]
        stage_counts = pd.Series(stages).value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=stage_counts.index,
            values=stage_counts.values,
            hole=0.62,
            marker=dict(colors=["#7c3aed", "#4f46e5", "#06b6d4", "#22c55e"],
                        line=dict(color=DARK_BG, width=2)),
            textfont=dict(color="#e2e8f0", size=11),
        ))
        fig_donut.add_annotation(
            text="Sleep<br>Stages", x=0.5, y=0.5, showarrow=False,
            font=dict(size=11, color="#94a3b8"), align="center"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8", size=10),
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.markdown('<div class="glass-card" style="text-align:center;padding:80px 0;color:#64748b;">Sleep distribution will appear here</div>',
                    unsafe_allow_html=True)

# ─── SIGNAL CHARTS ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Live Signal Streams</div>', unsafe_allow_html=True)

PLOT_CFG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,26,0.6)",
    font=dict(color="#94a3b8", size=11),
    height=220,
    margin=dict(l=40, r=20, t=36, b=30),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
               zeroline=False, color="#64748b"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
               zeroline=False, color="#64748b"),
)

sa1, sa2 = st.columns(2)

with sa1:
    emg_disp = emg_f[-400:]
    fig_emg = go.Figure()
    fig_emg.add_trace(go.Scatter(
        y=emg_disp, mode="lines",
        line=dict(color="#7c3aed", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
        name="EMG",
    ))
    fig_emg.update_layout(title="Filtered EMG Signal (last 2s)",
                          xaxis_title="Samples", yaxis_title="Voltage (V)",
                          **PLOT_CFG)
    st.plotly_chart(fig_emg, use_container_width=True)

with sa2:
    acc_disp = acc_f[-200:]
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        y=acc_disp, mode="lines",
        line=dict(color="#06b6d4", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(6,182,212,0.08)",
        name="Accel",
    ))
    fig_acc.update_layout(title="Acceleration Magnitude (filtered)",
                          xaxis_title="Samples", yaxis_title="m/s²",
                          **PLOT_CFG)
    st.plotly_chart(fig_acc, use_container_width=True)

# ─── STRESS HISTORY AREA CHART ────────────────────────────────────────────────
st.markdown('<div class="section-header">📉 Stress History</div>', unsafe_allow_html=True)

hist = list(st.session_state.stress_history)
if hist:
    fig_hist = go.Figure()
    # Zones
    n_h = len(hist)
    fig_hist.add_hrect(y0=0,    y1=0.30, fillcolor="rgba(34,197,94,0.06)",  line_width=0)
    fig_hist.add_hrect(y0=0.30, y1=0.65, fillcolor="rgba(245,158,11,0.06)", line_width=0)
    fig_hist.add_hrect(y0=0.65, y1=1.0,  fillcolor="rgba(239,68,68,0.06)",  line_width=0)

    colors = [CALM_GREEN if v < 0.3 else WARN_YELLOW if v < 0.65 else STRESS_RED for v in hist]
    fig_hist.add_trace(go.Scatter(
        y=hist, mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=colors, size=5, line=dict(color=DARK_BG, width=1)),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.1)",
        name="Stress %",
    ))
    # Threshold lines
    fig_hist.add_hline(y=0.30, line_dash="dash", line_color=CALM_GREEN,
                       line_width=1, opacity=0.5, annotation_text="Low / Moderate", annotation_font_size=9)
    fig_hist.add_hline(y=0.65, line_dash="dash", line_color=STRESS_RED,
                       line_width=1, opacity=0.5, annotation_text="Moderate / High", annotation_font_size=9)
    fig_hist.update_layout(
        xaxis_title="Epoch (1s intervals)",
        yaxis_title="Stress Probability",
        yaxis_range=[0, 1],
        **{**PLOT_CFG, "height": 260},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ─── BPM TREND ────────────────────────────────────────────────────────────────
if list(st.session_state.bpm_history):
    bpm_hist = list(st.session_state.bpm_history)
    fig_bpm = go.Figure()
    fig_bpm.add_trace(go.Scatter(
        y=bpm_hist, mode="lines",
        line=dict(color="#f472b6", width=2),
        fill="tozeroy",
        fillcolor="rgba(244,114,182,0.08)",
        name="BPM",
    ))
    fig_bpm.update_layout(
        title="Heart Rate Trend",
        xaxis_title="Epoch", yaxis_title="BPM",
        yaxis_range=[40, 140],
        **{**PLOT_CFG, "height": 200},
    )
    st.plotly_chart(fig_bpm, use_container_width=True)

# ─── SESSION SUMMARY TABLE ────────────────────────────────────────────────────
if st.session_state.session_log and len(st.session_state.session_log) >= 5:
    st.markdown('<div class="section-header">📋 Recent Session Log</div>', unsafe_allow_html=True)
    df_log = pd.DataFrame(st.session_state.session_log[-20:])
    df_log["stress_prob"] = (df_log["stress_prob"] * 100).round(1).astype(str) + "%"
    df_log["timestamp"] = df_log["timestamp"].str[11:19]
    df_log = df_log.rename(columns={
        "timestamp": "Time", "stress_prob": "Stress",
        "bpm": "BPM", "sleep_stage": "Stage",
        "emg_rms": "EMG RMS", "acc_std": "Acc Std"
    })
    st.dataframe(df_log[["Time", "Stress", "BPM", "Stage"]],
                 use_container_width=True, height=200)
