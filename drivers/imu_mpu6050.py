from mpu6050 import mpu6050


class MPU6050Driver:
    def __init__(self, addr=0x69):
        self.sensor = mpu6050(addr)

    def read_accel(self):
        return self.sensor.get_accel_data()
