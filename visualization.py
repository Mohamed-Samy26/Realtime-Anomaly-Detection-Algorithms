import matplotlib.pyplot as plt
import matplotlib.animation as animation

class RealTimeVisualizer:
    def __init__(self):
        """
        Initialize the real-time visualizer.
        """
        self.data = []
        self.anomalies = []
        self.fig, self.ax = plt.subplots()

    def update_plot(self, frame, stream, detector):
        """
        Update the plot in real time.
        - frame: The current frame (required for animation).
        - stream: The real-time data stream.
        - detector: The anomaly detection system.
        """
        data_point = stream.generate_data_point()
        is_anomaly = detector.is_anomaly(data_point)
        
        self.data.append(data_point)
        if is_anomaly:
            self.anomalies.append((len(self.data) - 1, data_point))

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

    def visualize(self, stream, detector, interval=1000):
        """
        Start real-time visualization of the data stream.
        - stream: The real-time data stream.
        - detector: The anomaly detection system.
        - interval: Update interval for the plot in milliseconds.
        """
        ani = animation.FuncAnimation(self.fig, self.update_plot, fargs=(stream, detector), interval=interval)
        plt.show()
