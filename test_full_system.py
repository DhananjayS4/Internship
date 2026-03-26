from sensor_service import SensorService
import time

sensor = SensorService()

while True:
    data = sensor.update()
    print(data)
    time.sleep(0.1)
