import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Cus_KMeans:
    def __init__(self, n_clusters=8, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centroids(self, data):
        indices = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        centroids = data[indices]
        return centroids

    def compute_distances(self, data):
        distances = np.sqrt(((data - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        return distances

    def assign_clusters(self, data):
        distances = self.compute_distances(data)
        cluster_labels = np.argmin(distances, axis=0)
        return cluster_labels

    def update_centroids(self, data, cluster_labels):
        centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def fit(self, X):
        scaled_data = StandardScaler().fit_transform(X)
        self.cluster_centers_ = self.initialize_centroids(scaled_data)
        for _ in range(self.max_iter):
            self.labels_ = self.assign_clusters(scaled_data)
            new_centroids = self.update_centroids(scaled_data, self.labels_)
            if np.allclose(self.cluster_centers_, new_centroids):
                break
            self.cluster_centers_ = new_centroids

    def predict(self, X):
        scaled_data = StandardScaler().fit_transform(X)
        distances = self.compute_distances(scaled_data)
        labels = np.argmin(distances, axis=0)
        return labels