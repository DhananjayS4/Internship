import smbus2
import time

class MAX30102:
    def __init__(self, address=0x57):
        self.bus = smbus2.SMBus(1)
        self.address = address

        # Reset
        self.bus.write_byte_data(self.address, 0x09, 0x40)
        time.sleep(1)

        # Mode: SpO2
        self.bus.write_byte_data(self.address, 0x09, 0x03)

        # LED config
        self.bus.write_byte_data(self.address, 0x0A, 0x27)
        self.bus.write_byte_data(self.address, 0x0C, 0x24)

    def read_fifo(self):
        data = self.bus.read_i2c_block_data(self.address, 0x07, 6)

        red = (data[0] << 16 | data[1] << 8 | data[2]) & 0x03FFFF
        ir = (data[3] << 16 | data[4] << 8 | data[5]) & 0x03FFFF

        return red, ir
