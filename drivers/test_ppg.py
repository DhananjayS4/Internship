from max30102_driver import MAX30102
import time

sensor = MAX30102()

print("Reading PPG...")

while True:
    red, ir = sensor.read_fifo()
    print(f"RED: {red} | IR: {ir}")
    time.sleep(0.1)
