import time

class Stopwatch:
    _start_time: float
    _curr_time: float
    _prev_time: float
    _dt: float
    
    def __init__(self):
        self._start_time = None  
        self._prev_time = 0.0    
        self._curr_time = 0.0    
        self._dt = 0.0           

    def start(self):
        """Start or restart the stopwatch."""
        self._start_time = time.perf_counter()
        self._prev_time = 0.0
        self._curr_time = 0.0
        self._dt = 0.0

    def sync(self):
        """Update the current time and delta time."""
        if self._start_time is None:
            self.start()  # Automatically start if not already started
            return

        # Calculate elapsed time relative to the start time
        self._prev_time = self._curr_time
        self._curr_time = time.perf_counter() - self._start_time
        self._dt = self._curr_time - self._prev_time