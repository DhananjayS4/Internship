from mpu6050 import mpu6050

sensor = mpu6050(0x69)  # change if needed
print(sensor.get_accel_data())
