import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution_timing.log'),
        logging.StreamHandler()
    ]
)

class TimingLogger:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger(name)

    def __enter__(self):
        self.start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"Starting {self.name} at {start_datetime}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Format duration
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        
        timing_str = f"Completed {self.name} in: "
        if hours > 0:
            timing_str += f"{hours}h "
        if minutes > 0:
            timing_str += f"{minutes}m "
        timing_str += f"{seconds:.2f}s"
        
        self.logger.info(timing_str)
        
        if exc_type is not None:
            self.logger.error(f"Error occurred: {exc_type.__name__}: {exc_val}")
            return False
        return True 