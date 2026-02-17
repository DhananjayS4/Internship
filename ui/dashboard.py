import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import time
from joblib import load
from collections import deque
import plotly.graph_objects as go

from drivers.emg_adc import EMGADC
from drivers.imu_mpu6050 import MPU6050Driver
from processing.filtering import bandpass_emg, lowpass_accel
from processing.features import extract_emg_features, extract_accel_features


# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "anxiety_model.joblib")

# ---------------- CONFIG ----------------
EMG_FS = 200
ACC_FS = 50
WINDOW_SEC = 30          # 30 sec epochs for sleep staging
HISTORY_LEN = 60

# ---------------- PAGE ----------------
st.set_page_config(page_title="Anxiety & Sleep Monitor", layout="wide")
st.title("🧠 Real-Time Anxiety & Sleep Monitoring System")
st.caption("EMG + Accelerometer based stress and sleep stage analysis")

# ---------------- SESSION STATE ----------------
if "emg_buffer" not in st.session_state:
    st.session_state.emg_buffer = []

if "acc_buffer" not in st.session_state:
    st.session_state.acc_buffer = []

if "stress_history" not in st.session_state:
    st.session_state.stress_history = deque(maxlen=HISTORY_LEN)

if "sleep_history" not in st.session_state:
    st.session_state.sleep_history = deque(maxlen=HISTORY_LEN)

# ---------------- LOAD MODEL ----------------
model = load(MODEL_PATH)

# ---------------- INIT SENSORS ----------------
emg = EMGADC()
mpu = MPU6050Driver()

# ---------------- DATA COLLECTION ----------------

for _ in range(EMG_FS * WINDOW_SEC):
    st.session_state.emg_buffer.append(emg.read_voltage())

for _ in range(ACC_FS * WINDOW_SEC):
    acc = mpu.read_accel()
    st.session_state.acc_buffer.append([
        acc["x"],
        acc["y"],
        acc["z"]
    ])

# Keep buffer sizes fixed
st.session_state.emg_buffer = st.session_state.emg_buffer[-EMG_FS*WINDOW_SEC:]
st.session_state.acc_buffer = st.session_state.acc_buffer[-ACC_FS*WINDOW_SEC:]

# Convert to numpy
emg_sig = np.array(st.session_state.emg_buffer)
acc_xyz = np.array(st.session_state.acc_buffer)   # (N,3)

# ---------------- SIGNAL PROCESSING ----------------
emg_f = bandpass_emg(emg_sig, EMG_FS)

acc_mag = np.linalg.norm(acc_xyz, axis=1)
acc_f = lowpass_accel(acc_mag, ACC_FS)

# ---------------- FEATURE EXTRACTION ----------------
f_emg = extract_emg_features(emg_f)
f_acc = extract_accel_features(acc_xyz)

feature_df = pd.DataFrame([{
    "emg_rms": f_emg["emg_rms"],
    "emg_var": f_emg["emg_var"],
    "emg_mean": f_emg["emg_mean"],
    "acc_mean": f_acc["acc_mean"],
    "acc_std": f_acc["acc_std"],
    "acc_max": f_acc["acc_max"],
}])

# ---------------- STRESS PREDICTION ----------------
stress_prob = model.predict_proba(feature_df)[0][1]
st.session_state.stress_history.append(stress_prob)

# ---------------- SLEEP STAGE CLASSIFIER ----------------
def classify_sleep_stage(emg_rms, acc_std, stress_prob):

    if acc_std > 0.4:
        return "AWAKE"

    if emg_rms < 2e-5 and acc_std < 0.05:
        return "DEEP SLEEP (N3)"

    if emg_rms < 3e-5 and stress_prob < 0.3:
        return "LIGHT SLEEP (N2)"

    return "REM SLEEP"


sleep_stage = classify_sleep_stage(
    f_emg["emg_rms"],
    f_acc["acc_std"],
    stress_prob
)

st.session_state.sleep_history.append(sleep_stage)

# ---------------- STATUS DISPLAY ----------------
col1, col2, col3 = st.columns(3)

with col1:
    if stress_prob < 0.3:
        st.success("🟢 LOW STRESS")
    elif stress_prob < 0.7:
        st.warning("🟡 MODERATE STRESS")
    else:
        st.error("🔴 HIGH STRESS")

with col2:
    st.metric("Stress Probability", f"{stress_prob*100:.2f}%")

with col3:
    st.info(f"😴 Sleep Stage: {sleep_stage}")

st.divider()

# ---------------- EMG GRAPH ----------------
fig_emg = go.Figure()
fig_emg.add_trace(go.Scatter(y=emg_f[-400:], mode="lines"))
fig_emg.update_layout(
    title="Filtered EMG Signal (Muscle Activity)",
    xaxis_title="Sample Index",
    yaxis_title="Voltage (V)",
    template="plotly_dark",
    height=300
)
st.plotly_chart(fig_emg, use_container_width=True)

# ---------------- ACC GRAPH ----------------
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(y=acc_f[-200:], mode="lines"))
fig_acc.update_layout(
    title="Filtered Acceleration Magnitude",
    xaxis_title="Sample Index",
    yaxis_title="Acceleration (m/s²)",
    template="plotly_dark",
    height=300
)
st.plotly_chart(fig_acc, use_container_width=True)

# ---------------- STRESS HISTORY ----------------
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    y=list(st.session_state.stress_history),
    mode="lines+markers"
))
fig_hist.update_layout(
    title="Stress Probability Over Time",
    xaxis_title="Epoch Index",
    yaxis_title="Probability (0–1)",
    template="plotly_dark",
    height=300
)
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------- AUTO REFRESH ----------------
time.sleep(1)
st.rerun()

