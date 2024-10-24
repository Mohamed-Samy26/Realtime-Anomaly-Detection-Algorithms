from algorithms import ZScoreAnomalyDetector, OnlineOneClassSVM
from data_generation import RealTimeDataStream
from visualization import RealTimeVisualizer
from test import OnlineKNNAnomalyDetector

# Create instances of the classes
stream = RealTimeDataStream(anomaly_prob=0.02, noise_level=0.1)
# detector = ZScoreAnomalyDetector(window_size=30, z_threshold=2.5)

detector = OnlineKNNAnomalyDetector(k=20, window_size=50, threshold=1.5)

visualizer = RealTimeVisualizer()

# Run the real-time visualization
visualizer.visualize(stream, detector, interval=10)  # 1-second interval
