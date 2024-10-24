from algorithms import ZScoreAnomalyDetector, OnlineKNNAnomalyDetector
from data_generation import RealTimeDataStream
from visualization import RealTimeVisualizer

# Create instances of the real-time data stream and anomaly detectors
stream = RealTimeDataStream(anomaly_prob=0.02, noise_level=0.1)

detector1 = ZScoreAnomalyDetector(window_size=50, z_threshold=2.3)
detector2 = OnlineKNNAnomalyDetector(k=9, window_size=20)


# Run the real-time visualization
visualizer = RealTimeVisualizer()
visualizer.visualize_multiple(stream, [detector1, detector2], interval=10)  # 10-Milli-Second interval
