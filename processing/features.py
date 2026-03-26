import numpy as np


# ── EMG ──────────────────────────────────────────────────────────────────────

def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def extract_emg_features(emg):
    return {
        "emg_rms":  float(rms(emg)),
        "emg_var":  float(np.var(emg)),
        "emg_mean": float(np.mean(np.abs(emg))),
    }


# ── Accelerometer ─────────────────────────────────────────────────────────────

def extract_accel_features(accel_xyz):
    """accel_xyz shape: (N, 3)"""
    mag = np.linalg.norm(accel_xyz, axis=1)
    return {
        "acc_mean": float(np.mean(mag)),
        "acc_std":  float(np.std(mag)),
        "acc_max":  float(np.max(mag)),
    }


# ── HRV ──────────────────────────────────────────────────────────────────────

def extract_hrv_features(rr_intervals):
    """
    Compute time-domain HRV metrics from RR intervals (in seconds).
    rr_intervals: list or 1-D array of RR durations
    """
    rr = np.array(rr_intervals, dtype=float)
    if len(rr) < 2:
        return {"hrv_rmssd": 0.0, "hrv_sdnn": 0.0, "hrv_pnn50": 0.0}

    diff_rr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    sdnn  = float(np.std(rr))
    pnn50 = float(np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr))
    return {"hrv_rmssd": rmssd, "hrv_sdnn": sdnn, "hrv_pnn50": pnn50}


# ── EDA ───────────────────────────────────────────────────────────────────────

def extract_eda_features(eda_signal):
    """Basic electrodermal activity features."""
    eda = np.array(eda_signal, dtype=float)
    return {"eda_mean": float(np.mean(eda))}
