"""
WESAD Dataset Loader
Loads raw physiological signals from WESAD .pkl files.

Dataset path: data/WESAD/WESAD/<SID>/<SID>.pkl

Label mapping:
    0 = not defined / transient
    1 = baseline (relaxed)
    2 = stress
    3 = amusement
    4 = meditation

We map: 1 -> 0 (relaxed), 2 -> 1 (stressed), others -> skip
"""

import pickle
import numpy as np
import os


# Sampling rates for chest (RespiBAN)
CHEST_FS = {
    "ACC":  700,
    "ECG":  700,
    "EMG":  700,
    "EDA":  700,
    "Temp": 700,
    "Resp": 700,
}

# Sampling rates for wrist (Empatica E4)
WRIST_FS = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}

LABEL_MAP = {1: 0, 2: 1}   # 1=baseline->relaxed, 2=stress->stressed


def load_subject(pkl_path: str) -> dict:
    """
    Load and return a single WESAD subject dictionary.

    Returns:
        {
          'chest': {'ACC': ndarray, 'ECG': ..., 'EMG': ..., 'EDA': ..., 'Temp': ..., 'Resp': ...},
          'wrist': {'ACC': ndarray, 'BVP': ..., 'EDA': ..., 'TEMP': ...},
          'label': ndarray (int8)
        }
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    return {
        "chest": raw["signal"]["chest"],
        "wrist": raw["signal"]["wrist"],
        "label": raw["label"].astype(np.int8),
    }


def get_subject_ids(wesad_root: str) -> list:
    """Return sorted list of subject IDs present in wesad_root."""
    ids = []
    for entry in os.scandir(wesad_root):
        if entry.is_dir() and entry.name.startswith("S"):
            pkl = os.path.join(entry.path, f"{entry.name}.pkl")
            if os.path.exists(pkl):
                ids.append(entry.name)
    return sorted(ids)


def get_pkl_path(wesad_root: str, subject_id: str) -> str:
    return os.path.join(wesad_root, subject_id, f"{subject_id}.pkl")
