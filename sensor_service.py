import time
import numpy as np

# Import your drivers
from drivers.max30102_driver import MAX30102
from drivers.imu_mpu6050 import MPU6050Driver
from drivers.emg_adc import EMGADC


class SensorService:
    def __init__(self):
        # Initialize sensors
        self.ppg = MAX30102()
        self.mpu = MPU6050Driver()
        self.emg = EMGADC()

        # Buffers
        from collections import deque
        self.BUFFER_SIZE = 200
        
        self.ppg_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.time_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.emg_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.acc_buffer = deque(maxlen=self.BUFFER_SIZE)

    # -------------------------
    # FEATURE FUNCTIONS
    # -------------------------

    def compute_emg_features(self):
        if len(self.emg_buffer) < 10:
            return 0

        signal = np.array(self.emg_buffer)
        rms = np.sqrt(np.mean(signal ** 2))
        return rms

    def compute_accel_features(self):
        if len(self.acc_buffer) < 10:
            return 0

        acc = np.array(self.acc_buffer)
        mag = np.sqrt(acc[:,0]**2 + acc[:,1]**2 + acc[:,2]**2)
        return np.std(mag)

    def compute_bpm_and_rr(self):
        signal = np.array(self.ppg_buffer)

        if len(signal) < 20:
            return 0, []

        # Guard: if signal is flat (sensor light off / disconnected),
        # normalization turns noise into fake peaks -> bogus BPM
        signal_range = np.max(signal) - np.min(signal)
        if signal_range < 500:   # raw ADC counts; adjust if using different units
            return 0, []

        # normalize
        signal = (signal - np.min(signal)) / (signal_range + 1e-6)

        peaks = []
        threshold = 0.5  # lowered from 0.6 for better sensitivity

        for i in range(1, len(signal)-1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)

        # filter peaks: minimum 0.4s gap (~150 BPM max)
        filtered = []
        min_gap = 0.4

        for p in peaks:
            if not filtered or (self.time_buffer[p] - self.time_buffer[filtered[-1]]) > min_gap:
                filtered.append(p)

        if len(filtered) < 2:
            return 0, []

        rr = []
        for i in range(1, len(filtered)):
            dt = self.time_buffer[filtered[i]] - self.time_buffer[filtered[i-1]]
            rr.append(dt)

        mean_rr = np.mean(rr)
        if mean_rr <= 0:
            return 0, []

        bpm = int(60 / mean_rr)

        # Physiological sanity check (30-220 BPM)
        if not (30 <= bpm <= 220):
            return 0, []

        return bpm, rr

    def compute_hrv(self, rr):
        if len(rr) < 2:
            return 0, 0

        rr = np.array(rr)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
        sdnn = np.std(rr)

        return rmssd, sdnn

    # -------------------------
    # MAIN LOOP
    # -------------------------

    def update(self):
        # Read sensors
        red, ir = self.ppg.read_fifo()
        accel = self.mpu.read_accel()
        emg_val = self.emg.read_voltage()

        # Only store PPG sample if sensor returned real data (0,0 = FIFO empty)
        if ir > 0:
            self.ppg_buffer.append(ir)
            self.time_buffer.append(time.time())

        self.emg_buffer.append(emg_val)
        self.acc_buffer.append([accel['x'], accel['y'], accel['z']])

        # Compute features
        bpm, rr = self.compute_bpm_and_rr()
        rmssd, sdnn = self.compute_hrv(rr)
        emg_rms = self.compute_emg_features()
        acc_std = self.compute_accel_features()

        return {
            "bpm": bpm,
            "rmssd": rmssd,
            "sdnn": sdnn,
            "emg_rms": emg_rms,
            "acc_std": acc_std
        }
