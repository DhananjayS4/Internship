import collections
import time


class SlidingWindow:
    def __init__(self, size_seconds, sample_rate):
        self.size = int(size_seconds * sample_rate)
        self.buffer = collections.deque(maxlen=self.size)
        self.start_time = time.time()

    def add(self, value):
        self.buffer.append(value)

    def is_full(self):
        return len(self.buffer) == self.size

    def get(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
