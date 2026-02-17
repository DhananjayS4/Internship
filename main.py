import time
import numpy as np
from joblib import load

from drivers.emg_adc import EMGADC
from drivers.imu_mpu6050 import MPU6050Driver

from processing.windowing import SlidingWindow
from processing.filtering import bandpass_emg, lowpass_accel
from processing.features import extract_emg_features, extract_accel_features


# ---------------- CONFIG ----------------
EMG_FS = 200
ACC_FS = 50
WINDOW_SEC = 3


# ---------------- LOAD MODEL ----------------
model = load("ml/models/anxiety_model.joblib")


# ---------------- INIT ------------------
emg = EMGADC()
mpu = MPU6050Driver()

emg_window = SlidingWindow(WINDOW_SEC, EMG_FS)
acc_window = SlidingWindow(WINDOW_SEC, ACC_FS)

print("Starting REAL-TIME stress prediction...")


# ---------------- LOOP ------------------
last_acc = time.time()
last_emg = time.time()

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
        emg_f = bandpass_emg(emg_sig, EMG_FS)

        acc_mag = np.linalg.norm(acc_sig, axis=1)
        acc_f = lowpass_accel(acc_mag, ACC_FS)

        # Features
        f_emg = extract_emg_features(emg_f)
        f_acc = extract_accel_features(acc_sig)

        features = np.array([[
            f_emg["emg_rms"],
            f_emg["emg_var"],
            f_emg["emg_mean"],
            f_acc["acc_mean"],
            f_acc["acc_std"],
            f_acc["acc_max"]
        ]])

        prediction = model.predict(features)[0]

        if prediction == 0:
            print("🟢 RELAXED")
        else:
            print("🔴 STRESSED")

        emg_window.clear()
        acc_window.clear()
import time
import numpy as np
from joblib import load

from drivers.emg_adc import EMGADC
from drivers.imu_mpu6050 import MPU6050Driver

from processing.windowing import SlidingWindow
from processing.filtering import bandpass_emg, lowpass_accel
from processing.features import extract_emg_features, extract_accel_features


# ---------------- CONFIG ----------------
EMG_FS = 200
ACC_FS = 50
WINDOW_SEC = 3


# ---------------- LOAD MODEL ----------------
model = load("ml/models/anxiety_model.joblib")


# ---------------- INIT ------------------
emg = EMGADC()
mpu = MPU6050Driver()

emg_window = SlidingWindow(WINDOW_SEC, EMG_FS)
acc_window = SlidingWindow(WINDOW_SEC, ACC_FS)

print("Starting REAL-TIME stress prediction...")


# ---------------- LOOP ------------------
last_acc = time.time()
last_emg = time.time()

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
        emg_f = bandpass_emg(emg_sig, EMG_FS)

        acc_mag = np.linalg.norm(acc_sig, axis=1)
        acc_f = lowpass_accel(acc_mag, ACC_FS)

        # Features
        f_emg = extract_emg_features(emg_f)
        f_acc = extract_accel_features(acc_sig)

        features = np.array([[
            f_emg["emg_rms"],
            f_emg["emg_var"],
            f_emg["emg_mean"],
            f_acc["acc_mean"],
            f_acc["acc_std"],
            f_acc["acc_max"]
        ]])

        prediction = model.predict(features)[0]

        if prediction == 0:
            print("🟢 RELAXED")
        else:
            print("🔴 STRESSED")

        emg_window.clear()
        acc_window.clear()

