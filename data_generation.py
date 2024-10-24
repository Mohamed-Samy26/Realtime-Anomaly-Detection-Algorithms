import numpy as np
import random
import time

class RealTimeDataStream:
    def __init__(self, anomaly_prob=0.01, noise_level=0.05):
        """
        Initialize the real-time data stream with configurable anomaly probability and noise level.
        
        Parameters:
        - anomaly_prob: Probability of an anomaly at any data point.
        - noise_level: Standard deviation of the random noise.
        """
        self.anomaly_prob = anomaly_prob
        self.noise_level = noise_level
        self.time_index = 0  # Tracks the progression of time in the stream

    def generate_data_point(self):
        """
        Generate the next data point in the stream based on a regular pattern, seasonality, and random noise.
        Anomalies may also be introduced.
        
        Returns:
        - data_point: The next data point in the stream.
        """
        # Regular pattern: linear trend and seasonal sine wave
        trend = 0.05 * self.time_index
        seasonality = 5 * np.sin(2 * np.pi * self.time_index / 50)
        
        # Random noise
        noise = np.random.normal(0, self.noise_level)
        
        # Base signal
        data_point = trend + seasonality + noise
        
        # Randomly introduce anomalies
        if random.random() < self.anomaly_prob:
            anomaly_value = random.choice([10, -10])  # Spike or dip
            data_point += anomaly_value
        
        # Increment time index for the next call
        self.time_index += 1
        
        return data_point

    def generate_data_point_tuple(self):
        return (self.generate_data_point(), self.time_index)

    def run(self, interval=1, points=10):
        """
        Run the data stream for a given number of points, yielding data in real time.
        
        Parameters:
        - interval: The time interval (in seconds) between data points.
        - points: The number of data points to generate.
        
        Yields:
        - data_point: The generated data point at each interval.
        """
        for _ in range(points):
            data_point = self.generate_data_point()
            print(f"New Data Point: {data_point}")
            time.sleep(interval)




# Example usage of the class

# Create a data stream object
# stream = RealTimeDataStream(anomaly_prob=0.02, noise_level=0.1)

# # Run the stream and get real-time data points
# stream.run(interval=1, points=10)
