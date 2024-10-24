
import numpy as np

class ConceptDriftDetection:
    def __init__(self, reference_data, bins=10):
        self.reference_data = reference_data
        self.bins = bins
        self.reference_hist, _ = np.histogram(reference_data, bins=self.bins, density=True)

    def calculate_psi(self, actual_data):
        actual_hist, _ = np.histogram(actual_data, bins=self.bins, density=True)
        # Avoid division by zero
        actual_hist = np.where(actual_hist == 0, 1e-10, actual_hist)
        psi = np.sum((actual_hist - self.reference_hist) * np.log(actual_hist / self.reference_hist))
        return psi

    def calculate_kld(self, actual_data):
        actual_hist, _ = np.histogram(actual_data, bins=self.bins, density=True)
        # Avoid division by zero
        actual_hist = np.where(actual_hist == 0, 1e-10, actual_hist)
        kld = np.sum(self.reference_hist * np.log(self.reference_hist / actual_hist))
        return kld


    def monitor_concept_drift(self, new_data):
        psi = self.calculate_psi(new_data)
        kld = self.calculate_kld(new_data)

        # Thresholds for drift detection
        if psi > 0.1:
            print(f"Concept drift detected with PSI: {psi:.4f}")
        if kld > 0.1:
            print(f"Concept drift detected with KLD: {kld:.4f}")

        return psi, kld

# # Example usage
# reference_data = np.random.normal(0, 1, 1000)  # Initial reference data
# ad = ConceptDriftDetection(reference_data)

# # Simulating new incoming data
# new_data = np.random.normal(0.5, 1, 100)  # New data simulating a shift
# ad.monitor_concept_drift(new_data)
# # anomalies = ad.detect_anomalies(new_data)

# print("Anomaly detection results:", anomalies)
