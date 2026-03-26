import time
import csv
import numpy as np
from drivers.emg_adc import EMGADC
from drivers.imu_mpu6050 import MPU6050Driver

emg = EMGADC()
mpu = MPU6050Driver()

with open("data/live_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["emg", "ax", "ay", "az"])

    while True:
        v = emg.read_voltage()
        acc = mpu.read_accel()

        writer.writerow([v, acc["x"], acc["y"], acc["z"]])
        f.flush()
        time.sleep(0.02)   # 50 Hz stable
