import numpy as np
from collections import deque

class ZScoreAnomalyDetector:
    def __init__(self, window_size:int = 50, z_threshold:int = 3):
        """
        Initialize the anomaly detection class.
        - window_size: Size of the sliding window for moving average.
        - z_threshold: Threshold for Z-score-based anomaly detection.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data_history = deque(maxlen=window_size)

    def is_anomaly(self, data_point:float) -> bool:
        """
        Detect anomalies using Z-score method.
        """
        self.data_history.append(data_point)
        
        if len(self.data_history) < self.window_size:
            # print("Warning: Not enough data to calculate statistics.")
            return False
        
        mean = np.mean(self.data_history)
        std_dev = np.std(self.data_history)
        
        if std_dev == 0:
            return False  # Avoid division by zero
        
        z_score = (data_point - mean) / std_dev
        return abs(z_score) > self.z_threshold
    
