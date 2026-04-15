import time
import numpy as np
from joblib import load

from drivers.emg_adc import EMGADC
from drivers.imu_mpu6050 import MPU6050Driver

from processing.windowing import SlidingWindow
from processing.filtering import bandpass_emg, lowpass_accel
from processing.features import extract_emg_features, extract_accel_features


# ---------------- CONFIG ----------------
EMG_FS     = 200
ACC_FS     = 50
WINDOW_SEC = 3

# Stress threshold: raise this if it always says STRESSED (try 0.70, 0.75)
# Lower it if it never detects stress (try 0.55)
STRESS_THRESHOLD = 0.65


# ---------------- LOAD MODEL ----------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_model_v2 = os.path.join(BASE_DIR, "ml", "models", "anxiety_model_v2.joblib")
_model_v1 = os.path.join(BASE_DIR, "ml", "models", "anxiety_model.joblib")
model = load(_model_v2 if os.path.exists(_model_v2) else _model_v1)

# Detect how many features the model expects
try:
    N_FEATURES = model.n_features_in_
except AttributeError:
    try:
        N_FEATURES = model.steps[-1][1].n_features_in_
    except Exception:
        N_FEATURES = 6

has_proba = hasattr(model, "predict_proba")
print(f"Model loaded. Expects {N_FEATURES} features | predict_proba: {has_proba}")
print(f"Stress threshold: {STRESS_THRESHOLD}")


# ---------------- INIT ------------------
emg = EMGADC()
mpu = MPU6050Driver()

emg_window = SlidingWindow(WINDOW_SEC, EMG_FS)
acc_window = SlidingWindow(WINDOW_SEC, ACC_FS)

print("Starting REAL-TIME stress prediction...")
print("-" * 50)

# ---------------- LOOP ------------------
last_acc   = time.time()
last_emg   = time.time()
debug_count = 0          # print feature values for first 5 windows

while True:
    now = time.time()

    if now - last_emg >= 1 / EMG_FS:
        v = emg.read_voltage()
        emg_window.add(v)
        last_emg = now

    if now - last_acc >= 1 / ACC_FS:
        acc = mpu.read_accel()
        acc_window.add([acc["x"], acc["y"], acc["z"]])
        last_acc = now

    if emg_window.is_full() and acc_window.is_full():

        emg_sig = np.array(emg_window.get())
        acc_sig = np.array(acc_window.get())

        # Filtering
        emg_f  = bandpass_emg(emg_sig, EMG_FS)
        acc_mag = np.linalg.norm(acc_sig, axis=1)
        acc_f  = lowpass_accel(acc_mag, ACC_FS)

        # Features
        f_emg = extract_emg_features(emg_f)
        f_acc = extract_accel_features(acc_sig)

        feature_row = [
            f_emg["emg_rms"],
            f_emg["emg_var"],
            f_emg["emg_mean"],
            f_acc["acc_mean"],
            f_acc["acc_std"],
            f_acc["acc_max"]
        ]

        # If model expects 10 features (WESAD), pad with neutral HRV/EDA values.
        if N_FEATURES == 10:
            feature_row += [0.04, 0.05, 0.30, 2.0]  # hrv_rmssd, hrv_sdnn, hrv_pnn50, eda_mean

        features = [feature_row]

        # Debug: print feature values for first 5 windows
        if debug_count < 5:
            print(f"[DEBUG window {debug_count+1}]")
            print(f"  emg_rms={f_emg['emg_rms']:.6f}  emg_var={f_emg['emg_var']:.8f}  emg_mean={f_emg['emg_mean']:.6f}")
            print(f"  acc_mean={f_acc['acc_mean']:.4f}  acc_std={f_acc['acc_std']:.4f}  acc_max={f_acc['acc_max']:.4f}")
            debug_count += 1

        # Use predict_proba if available for adjustable threshold
        if has_proba:
            stress_prob = float(model.predict_proba(features)[0][1])
            stressed = stress_prob >= STRESS_THRESHOLD
            label = "STRESSED" if stressed else "RELAXED"
            icon  = "\U0001f534" if stressed else "\U0001f7e2"
            print(f"{icon} {label}  (stress prob: {stress_prob:.2f}  threshold: {STRESS_THRESHOLD})")
        else:
            # Fallback to hard predict
            prediction = model.predict(features)[0]
            print("\U0001f7e2 RELAXED" if prediction == 0 else "\U0001f534 STRESSED")

        emg_window.clear()
        acc_window.clear()
