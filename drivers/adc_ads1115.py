import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class ADS1115Driver:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(i2c)

        # Use channel 0 correctly
        self.channel = AnalogIn(self.ads, 0)

    def read(self):
        return self.channel.voltage

