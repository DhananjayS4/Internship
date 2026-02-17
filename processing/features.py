import numpy as np


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def extract_emg_features(emg):
    return {
        "emg_rms": rms(emg),
        "emg_var": np.var(emg),
        "emg_mean": np.mean(emg),
    }


def extract_accel_features(accel_xyz):
    mag = np.linalg.norm(accel_xyz, axis=1)

    return {
        "acc_mean": np.mean(mag),
        "acc_std": np.std(mag),
        "acc_max": np.max(mag),
    }
