# src/utils/timer.py
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        print(f"{self.duration:.4f} seconds")