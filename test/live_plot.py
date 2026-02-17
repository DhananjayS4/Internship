import time
import collections
import board, busio
import matplotlib.pyplot as plt

import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from mpu6050 import mpu6050

BUFFER = 300

i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)
emg = AnalogIn(ads, 0)

mpu = mpu6050(0x69)

emg_buf = collections.deque(maxlen=BUFFER)
acc_buf = collections.deque(maxlen=BUFFER)

plt.ion()
fig, ax = plt.subplots()

while True:
    emg_buf.append(emg.voltage)

    acc = mpu.get_accel_data()
    mag = (acc["x"]**2 + acc["y"]**2 + acc["z"]**2) ** 0.5
    acc_buf.append(mag)

    ax.clear()
    ax.plot(emg_buf)
    ax.plot(acc_buf)
    plt.pause(0.01)
