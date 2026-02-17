import numpy as np
from scipy.signal import butter, filtfilt


# Bandpass filter for EMG (20–450 Hz typical)
def bandpass_emg(signal, fs, low=10, high=90, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)



# Low-pass filter for accelerometer magnitude (<15 Hz)
def lowpass_accel(signal, fs, cutoff=15, order=4):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)
