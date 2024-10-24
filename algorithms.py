from typing import Tuple
import numpy as np
from collections import deque

# abstract class for anomaly detectors
class AnomalyDetector:
    def is_anomaly(self, data_point:Tuple[int, int]) -> bool:
        """
        Detect anomalies based on the given data point.
        """
        raise NotImplementedError
    
    def get_name(self) -> str:
        """
        Get the name of the anomaly detector.
        """
        return self.__class__.__name__

class ZScoreAnomalyDetector(AnomalyDetector):
    def __init__(self, window_size:int = 50, z_threshold:int = 3):
        """
        Initialize the anomaly detection class.
        - window_size: Size of the sliding window for moving average.
        - z_threshold: Threshold for Z-score-based anomaly detection.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data_history = deque(maxlen=window_size)

    def is_anomaly(self, data_point:Tuple[float, float]) -> bool:
        """
        Detect anomalies using Z-score method.
        return: True if the data point is an anomaly, False otherwise.
        """
        # Add the new data point to the history sliding window
        self.data_history.append(data_point)
        
        if len(self.data_history) < self.window_size:
            # Not enough data to calculate statistics
            return False
        
        # Calculate the mean and standard deviation of the data history
        data_array = np.array(self.data_history)
        mean = np.mean(data_array, axis=0)
        std_dev = np.std(data_array, axis=0)
        
        if np.any(std_dev == 0):
            return False  # Avoid division by zero
        
        # Calculate the Z-scores for the data point
        z_scores = (data_point - mean) / std_dev
        is_anomaly = np.any(np.abs(z_scores) > self.z_threshold)
        
        # remove anomalous data points and slide the window
        if is_anomaly:
            self.data_history.pop()
        else:        
            self.data_history.popleft()
        
        return is_anomaly

class OnlineKNNAnomalyDetector(AnomalyDetector):
    def __init__(self, k=3, window_size=100, threshold=None):
        """
        Initialize the online K-Nearest Neighbors anomaly detection class.
        - k: Number of nearest neighbors to consider.
        - window_size: Maximum number of points to keep in memory.
        - threshold: Threshold for anomaly detection, will be auto-calculated if None.
        """
        if k < 1 or window_size < k:
            raise ValueError("Invalid k or window_size, K must be greater than 0 and window_size must be greater than k")
        self.k = k  # Number of nearest neighbors
        self.window_size = window_size  # Maximum number of points to keep in memory
        self.memory = np.empty((0, 2))  # Initialize an empty memory
        self.is_auto_threshold_calculated = False
        self.threshold = threshold  # Default threshold, will be auto-calculated if None

    def add_point(self, point: Tuple[float, float])->None:
        """
        Add a new data point to the memory.
        """
        # Add the new point to memory
        if len(self.memory) >= self.window_size:
            self.memory = np.delete(self.memory, 0, axis=0)  # Remove the oldest point if memory is full
        self.memory = np.vstack([self.memory, point])
        
        if self.threshold is None and not self.is_auto_threshold_calculated and len(self.memory) >= self.window_size:
            self.threshold = self.calculate_threshold()
            self.is_auto_threshold_calculated = True
            print(f"Auto-calculated threshold: {self.threshold}")

    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float])->float:
        """
        Calculate the Euclidean distance between two points.
        return: The Euclidean distance between the two points.
        """
        # Euclidean distance between two points
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def calculate_threshold(self)->float:
        """
        Calculate the threshold for anomaly detection based on the current memory.
        based on > 99% confidence interval.
        return: The calculated threshold.
        """
        # Calculate pairwise distances between all points in memory
        distances = []
        for i, point in enumerate(self.memory):
            other_points = np.delete(self.memory, i, axis=0)  # Exclude the current point
            dists = np.array([self.distance(point, other_point) for other_point in other_points])
            dists = np.sort(dists)
            knn_distance = dists[self.k - 1]  # Take the k-th nearest distance
            distances.append(knn_distance)

        # Use mean + some factor of standard deviation as the threshold
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        return mean_distance + (2.7 * std_distance)  # > 99% confidence interval

    def is_anomaly(self, point: Tuple[float, float])->bool:
        """
        Detect anomalies based on the given data point.
        return: True if the data point is an anomaly, False otherwise.
        """
        
        if len(self.memory) < self.window_size:
            self.add_point(point)
            return False  # Not enough points to detect anomaly
        
        # Calculate distances to all points in memory
        distances = np.array([self.distance(point, mem_point) for mem_point in self.memory])
        distances = np.sort(distances)
        
        # Take the k-th nearest distance
        knn_distance = distances[self.k - 1]

        # If distance exceeds the threshold, it's an anomaly
        # print(f"KNN distance: {knn_distance}, Threshold: {self.threshold}")
        
        is_anomaly =  knn_distance > self.threshold
        
        # add the point to memory if not an anomaly
        if not is_anomaly:
            self.add_point(point)
            
        return is_anomaly
