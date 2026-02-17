import board, busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

class EMG_ADC:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS1115(self.i2c)
        self.chan = AnalogIn(self.ads, ADS1115.P0)

    def read(self):
        return self.chan.voltage


