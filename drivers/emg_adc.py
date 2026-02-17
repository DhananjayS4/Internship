import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class EMGADC:
    def __init__(self, channel=0):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(self.i2c)
        self.chan = AnalogIn(self.ads, channel)

    def read_voltage(self):
        return self.chan.voltage
