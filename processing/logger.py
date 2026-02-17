import csv, time

class DataLogger:
    def __init__(self, filename):
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["t","emg","ax","ay","az"])
        self.start = time.time()

    def log(self, emg, acc):
        t = time.time() - self.start
        self.writer.writerow([t, emg, acc["x"], acc["y"], acc["z"]])

    def close(self):
        self.file.close()
