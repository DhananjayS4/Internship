from mpu6050 import mpu6050
import time

sensor = mpu6050(0x68)  # Use the address we saw earlier

while True:
    data = sensor.get_accel_data()
    print(data)
    time.sleep(0.5)

