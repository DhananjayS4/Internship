import time
import board
import busio

from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)

# Channel 0
chan = AnalogIn(ads, 0)

while True:
    print("Voltage:", chan.voltage)
    time.sleep(0.1)

