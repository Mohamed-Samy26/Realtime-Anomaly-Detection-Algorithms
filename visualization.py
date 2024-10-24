from typing import List
import matplotlib.pyplot as plt, matplotlib.cm as cm
import matplotlib.animation as animation
from algorithms import AnomalyDetector
from data_generation import RealTimeDataStream

class RealTimeVisualizer:
    def __init__(self):
        """
        Initialize the real-time visualizer.
        """
        self.data = []
        self.anomalies = []
        self.fig, self.ax = plt.subplots()

    def update_plot(self, frame, stream:RealTimeDataStream, detector:AnomalyDetector):
        """
        Update the plot in real time.
        - frame: The current frame (required for animation).
        - stream: The real-time data stream.
        - detector: The anomaly detection system.
        """
        data_point_tuple = stream.generate_data_point_tuple()
        is_anomaly = detector.is_anomaly(data_point_tuple)
        
        self.data.append(data_point_tuple[0])
        if is_anomaly:
            self.anomalies.append((data_point_tuple[1], data_point_tuple[0]))

        self.ax.clear()
        self.ax.plot(self.data, label='Data Stream')
        
        # Highlight anomalies
        if self.anomalies:
            anomaly_indices, anomaly_points = zip(*self.anomalies)
            self.ax.scatter(anomaly_indices, anomaly_points, color='red', label='Anomalies')
        
        self.ax.legend()
        self.ax.set_title('Real-Time Data Stream with Anomalies')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')

    def visualize(self, stream:RealTimeDataStream, detector:AnomalyDetector, interval:int=1000):
        """
        Start real-time visualization of the data stream.
        - stream: The real-time data stream.
        - detector: The anomaly detection system.
        - interval: Update interval for the plot in milliseconds.
        """
        ani = animation.FuncAnimation(self.fig, self.update_plot, fargs=(stream, detector), interval=interval)
        plt.show()
        

    def update_plot_multiple(self, frame, stream: RealTimeDataStream, detectors: List[AnomalyDetector], colors: List[str]):
        """
        Update the plot in real time for multiple detectors, marking anomalies in different colors.
        - frame: The current frame (required for animation).
        - stream: The real-time data stream.
        - detectors: List of anomaly detection systems.
        - colors: List of colors for marking anomalies from each detector.
        """
        data_point_tuple = stream.generate_data_point_tuple()
        self.data.append(data_point_tuple[0])

        # Process anomalies for each detector and store them with a unique color
        for i, detector in enumerate(detectors):
            is_anomaly = detector.is_anomaly(data_point_tuple)
            if is_anomaly:
                if len(self.anomalies_by_detector) <= i:
                    self.anomalies_by_detector.append([])  # Initialize anomaly list for this detector
                self.anomalies_by_detector[i].append((len(self.data) - 1, data_point_tuple[0]))

        # Clear and re-plot the data stream
        self.ax.clear()
        self.ax.plot(self.data, label='Data Stream')

        # Plot anomalies with different colors and detector names
        for i, anomalies in enumerate(self.anomalies_by_detector):
            if anomalies:
                anomaly_indices, anomaly_points = zip(*anomalies)
                detector_name = detectors[i].get_name()  # Get detector name
                self.ax.scatter(anomaly_indices, anomaly_points, color=colors[i], label=f'{detector_name} Anomalies')

        self.ax.legend()
        self.ax.set_title('Real-Time Data Stream with Multiple Anomalies Detected')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')    

    def visualize_multiple(self, stream: RealTimeDataStream, detectors: List[AnomalyDetector], interval: int = 1000):
        """
        Start real-time visualization of the data stream with multiple detectors.
        Automatically assigns distinct colors to each detector.
        - stream: The real-time data stream.
        - detectors: List of anomaly detection systems.
        - interval: Update interval for the plot in milliseconds.
        """
        # Automatically generate distinct colors for each detector
        num_detectors = len(detectors)
        cmap = cm.get_cmap('tab20b', num_detectors)  # colormap for distinct colors
        colors = [cmap(i) for i in range(num_detectors)]

        # Initialize anomaly lists for each detector
        self.anomalies_by_detector = [[] for _ in detectors]

        ani = animation.FuncAnimation(self.fig, self.update_plot_multiple, fargs=(stream, detectors, colors), interval=interval)
        
        # change window title
        self.fig.canvas.set_window_title('Mohamed Samy: Real-Time Data Stream with Multiple Anomalies Detected')
        plt.show()