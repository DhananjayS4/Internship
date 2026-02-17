import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import time

from drivers.emg_adc import EMGADC
from drivers.imu_mpu6050 import MPU6050Driver

emg = EMGADC()
mpu = MPU6050Driver()

print("Reading from DRIVERS layer...")

while True:
    v = emg.read_voltage()
    acc = mpu.read_accel()

    print(
        f"EMG={v:.3f}V | "
        f"AX={acc['x']:.2f} "
        f"AY={acc['y']:.2f} "
        f"AZ={acc['z']:.2f}"
    )

    time.sleep(0.05)
