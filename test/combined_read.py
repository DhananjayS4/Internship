import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from mpu6050 import mpu6050

# I2C
i2c = busio.I2C(board.SCL, board.SDA)

# ADC / EMG
ads = ADS.ADS1115(i2c)
emg = AnalogIn(ads, 0)

# MPU6050 (use detected address)
mpu = mpu6050(0x69)

print("Streaming EMG + Accelerometer...")

while True:
    emg_val = emg.voltage
    acc = mpu.get_accel_data()

    print(
        f"EMG={emg_val:.3f}V | "
        f"AX={acc['x']:.2f} "
        f"AY={acc['y']:.2f} "
        f"AZ={acc['z']:.2f}"
    )

    time.sleep(0.05)
