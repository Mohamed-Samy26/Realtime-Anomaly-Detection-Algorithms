import numpy as np

class OnlineKNNAnomalyDetector:
    def __init__(self, k=3, window_size=100, threshold=1.5):
        self.k = k  # Number of nearest neighbors
        self.window_size = window_size  # Maximum number of points to keep in memory
        self.memory = np.empty((0, 2))  # Initialize an empty memory
        self.threshold = threshold  # Threshold for anomaly detection

    def add_point(self, point):
        # Add the new point to memory
        if len(self.memory) >= self.window_size:
            self.memory = np.delete(self.memory, 0, axis=0)  # Remove the oldest point if memory is full
        self.memory = np.vstack([self.memory, point])

    def distance(self, p1, p2):
        # Euclidean distance between two points
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def is_anomaly(self, point):
        self.add_point(point)  # Add the new point to memory
        if len(self.memory) < self.k:
            return False  # Not enough points to detect anomaly
        
        # Calculate distances to all points in memory
        distances = np.array([self.distance(point, mem_point) for mem_point in self.memory])
        distances = np.sort(distances)
        
        # Take the k-th nearest distance
        knn_distance = distances[self.k - 1]

        # If distance exceeds the threshold, it's an anomaly
        return knn_distance > self.threshold

# Example usage:
detector = OnlineKNNAnomalyDetector(k=3, window_size=100)
data_points = np.random.normal(0, 1, (100, 2))  # 100 normal points (non-anomalous)
anomalies = []

# Simulate streaming data
for point in data_points:
    detector.add_point(point)
    if detector.is_anomaly(point, threshold=1.5):
        anomalies.append(point)

# print(f"Detected anomalies: {len(anomalies)}")
