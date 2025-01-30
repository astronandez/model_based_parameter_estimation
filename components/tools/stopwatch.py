import time

class Stopwatch:
    start_time: float
    curr_time: float
    prev_time: float
    dt: float
    
    def __init__ (self, start_time: float = None):
        self.start_time = start_time
    
    def sync(self):
        try:
            self.prev_time = self.curr_time
            self.curr_time = time.time() - self.start_time
            self.dt = self.curr_time - self.prev_time
        except:
            self.start_time = time.time()
            self.curr_time = self.start_time - self.start_time
            self.prev_time = 0.0
            self.dt = 0.0