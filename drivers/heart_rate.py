from max30102_driver import MAX30102
import time
import numpy as np

sensor = MAX30102()

buffer = []
time_buffer = []
BUFFER_SIZE = 200

def calculate_bpm(signal, times):
    signal = np.array(signal)

    # normalize
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

    peaks = []
    threshold = 0.6

    for i in range(1, len(signal)-1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)

    # filter close peaks
    filtered = []
    min_gap = 0.4  # seconds

    for p in peaks:
        if not filtered or (times[p] - times[filtered[-1]]) > min_gap:
            filtered.append(p)

    if len(filtered) < 2:
        return 0

    intervals = []
    for i in range(1, len(filtered)):
        dt = times[filtered[i]] - times[filtered[i-1]]
        intervals.append(dt)

    bpm = 60 / np.mean(intervals)

    return int(bpm)

print("Calculating Heart Rate...")

while True:
    red, ir = sensor.read_fifo()

    buffer.append(ir)
    time_buffer.append(time.time())

    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)
        time_buffer.pop(0)

        bpm = calculate_bpm(buffer, time_buffer)

        print(f"BPM: {bpm}")

    time.sleep(0.05)
