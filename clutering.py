import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans as SKLearnKMeans
import random

class KMeans:
    def __init__(self, k=2):
        self.k = k
        self.centroids = []
        

    def fit(self, X, max_iterations=10000): 
        # Randomly initialize centroids
        self.centroids = np.array(random.sample(X.tolist(), self.k))

        for _ in range(max_iterations):
            # Assign each data point to the nearest centroid
            clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = np.linalg.norm(self.centroids - x, axis=1)
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(x)

            # Update centroids
            new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = np.array(new_centroids)

    def predict(self, X):
        distances = np.linalg.norm(self.centroids - X[:, np.newaxis], axis=2)
        return np.argmin(distances, axis=1)
        

# Load the Iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target

# Apply custom KMeans clustering
custom_kmeans = KMeans(3)  # Assuming we want to cluster into 3 groups
custom_kmeans.fit(X)
y_custom_pred = custom_kmeans.predict(X)

# Apply sklearn KMeans clustering
sklearn_kmeans = SKLearnKMeans(n_clusters=3, random_state=42)
y_sklearn_pred = sklearn_kmeans.fit_predict(X)

# Evaluate clustering using Adjusted Rand Index
ari_custom = adjusted_rand_score(y_true, y_custom_pred)
ari_sklearn = adjusted_rand_score(y_true, y_sklearn_pred)

print("Adjusted Rand Index (Custom KMeans):", ari_custom)
print("Adjusted Rand Index (sklearn KMeans):", ari_sklearn)
