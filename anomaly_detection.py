import numpy as np
import pandas as pd

class AnomalyDetection:
    def __init__(self):
        pass
    
    def fit(self, X):
        probabilities = self.PDF(X)
        # Product of the probabilities may result in underflow, so we take the average of the log probabilities
        log_probabilities = np.log(probabilities)
        average_log_probabilities = np.mean(log_probabilities, axis=1)
        # Choose a threshold dynamically based on the distribution of log probabilities
        threshold = np.percentile(average_log_probabilities, 0.7)
        anomalies = np.where(average_log_probabilities <= threshold)
        return X[anomalies]
        
    def mean(self, X):
        return np.mean(X, axis=0)
    
    def std(self, X):
        return np.std(X, axis=0)
    
    def PDF(self, X):
        # Mean and std for each feature
        mean = self.mean(X)
        std = self.std(X)
        # Probability density function for a Gaussian
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((X - mean) ** 2 / (2 * std ** 2)))

# Load your data
data = pd.read_csv('anomaly_detection_sample_data.csv')

# Extract the features
X = data.iloc[:, :-1].values

detector = AnomalyDetection()
anomalies = detector.fit(X)
print(anomalies)
