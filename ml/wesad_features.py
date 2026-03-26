"""
WESAD Feature Extractor
Extracts windowed features from raw WESAD signals compatible with
the EASD model input schema, plus enriched HRV and EDA features.

Feature vector (10 features):
    emg_rms, emg_var, emg_mean,
    acc_mean, acc_std, acc_max,
    hrv_rmssd, hrv_sdnn, hrv_pnn50,
    eda_mean
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ───────────────────────── helpers ──────────────────────────────────────

def _butter_bandpass(signal, fs, low, high, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def _butter_lowpass(signal, fs, cutoff, order=4):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


# ───────────────────────── EMG ──────────────────────────────────────────

def emg_features(emg: np.ndarray, fs: int = 700) -> dict:
    # Filter 20–450 Hz
    try:
        emg_f = _butter_bandpass(emg, fs, 20, min(450, fs * 0.49))
    except Exception:
        emg_f = emg
    return {
        "emg_rms":  float(np.sqrt(np.mean(emg_f ** 2))),
        "emg_var":  float(np.var(emg_f)),
        "emg_mean": float(np.mean(np.abs(emg_f))),
    }


# ───────────────────────── Accelerometer ────────────────────────────────

def accel_features(acc: np.ndarray) -> dict:
    """acc shape: (N, 3)"""
    mag = np.linalg.norm(acc, axis=1)
    return {
        "acc_mean": float(np.mean(mag)),
        "acc_std":  float(np.std(mag)),
        "acc_max":  float(np.max(mag)),
    }


# ───────────────────────── HRV from BVP ─────────────────────────────────

def hrv_features_from_bvp(bvp: np.ndarray, fs: int = 64) -> dict:
    """Compute HRV metrics from BVP signal via peak detection."""
    try:
        # Smooth BVP
        bvp_f = _butter_lowpass(bvp, fs, cutoff=8)
        # Normalise
        bvp_n = (bvp_f - np.min(bvp_f)) / (np.max(bvp_f) - np.min(bvp_f) + 1e-9)
        # Detect peaks with min distance ~0.4s
        peaks, _ = find_peaks(bvp_n, height=0.4, distance=int(fs * 0.4))

        if len(peaks) < 3:
            return {"hrv_rmssd": 0.0, "hrv_sdnn": 0.0, "hrv_pnn50": 0.0}

        rr = np.diff(peaks) / fs          # RR intervals in seconds
        diff_rr = np.diff(rr)

        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
        sdnn  = float(np.std(rr))
        pnn50 = float(np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr))

    except Exception:
        rmssd, sdnn, pnn50 = 0.0, 0.0, 0.0

    return {"hrv_rmssd": rmssd, "hrv_sdnn": sdnn, "hrv_pnn50": pnn50}


# ───────────────────────── EDA ──────────────────────────────────────────

def eda_features(eda: np.ndarray) -> dict:
    return {"eda_mean": float(np.mean(eda))}


# ───────────────────────── Window extractor ─────────────────────────────

def extract_windows(subject_data: dict,
                    window_sec: int = 60,
                    overlap: float = 0.5) -> list[dict]:
    """
    Slide a window across the label-aligned chest signals.
    Returns a list of feature-row dicts with 'label' key.
    """
    from ml.wesad_loader import CHEST_FS, WRIST_FS, LABEL_MAP

    label     = subject_data["label"]           # aligned to 700 Hz chest
    chest     = subject_data["chest"]
    wrist     = subject_data["wrist"]

    fs_chest  = CHEST_FS["EMG"]                 # 700 Hz
    fs_bvp    = WRIST_FS["BVP"]                 # 64 Hz
    fs_wrist_acc = WRIST_FS["ACC"]              # 32 Hz

    win       = int(window_sec * fs_chest)
    step      = int(win * (1 - overlap))

    rows = []
    n = len(label)

    i = 0
    while i + win <= n:
        seg_label = label[i: i + win]

        # Use majority label; skip if not clean stress/baseline
        unique, counts = np.unique(seg_label, return_counts=True)
        majority = unique[np.argmax(counts)]
        purity = np.max(counts) / win

        if majority not in LABEL_MAP or purity < 0.8:
            i += step
            continue

        y = LABEL_MAP[majority]

        # ── Chest signals (700 Hz) ──────────────────────────────────
        emg_seg  = chest["EMG"][i: i + win].flatten()
        acc_seg  = chest["ACC"][i: i + win]          # (N, 3)
        eda_seg  = chest["EDA"][i: i + win].flatten()

        # ── Wrist BVP (64 Hz) — scale indices ──────────────────────
        ratio_bvp = fs_bvp / fs_chest
        j_bvp     = int(i * ratio_bvp)
        k_bvp     = int((i + win) * ratio_bvp)
        bvp_seg   = wrist["BVP"][j_bvp: k_bvp].flatten()

        # ── Feature extraction ──────────────────────────────────────
        feat = {}
        feat.update(emg_features(emg_seg, fs=fs_chest))
        feat.update(accel_features(acc_seg))
        feat.update(hrv_features_from_bvp(bvp_seg, fs=fs_bvp))
        feat.update(eda_features(eda_seg))
        feat["label"] = y

        rows.append(feat)
        i += step

    return rows
