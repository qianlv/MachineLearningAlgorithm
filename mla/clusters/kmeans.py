# -*- coding: utf-8 -*-
import numpy as np
import random
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


class KMeans(object):
    def __init__(self, k):
        self.k = k
        self.clusters = None
        self.centroids = None

    def predict(self, X, max_iters=300):
        X = np.atleast_2d(X)
        n_samples, _ = np.shape(X)
        choice = random.sample(range(0, n_samples), self.k)
        self.centroids = X[choice]
        self.centroids = np.atleast_2d(self.centroids)
        it = 0
        for _ in range(max_iters):
            it += 1
            self.clusters = [set() for _ in range(self.k)]
            self._nearest(X)
            old_centroid = self._update_centroids(X)
            #  self.plot(self._get_predict(n_samples), X)
            if self._is_stop(old_centroid, self.centroids):
                break
        print("Iterator: {0}".format(it))
        return self._get_predict(n_samples)

    def _get_predict(self, n_samples):
        preds = np.zeros(n_samples)
        for label, clusters in enumerate(self.clusters):
            for i in clusters:
                preds[i] = label
        return preds

    def _nearest(self, X):
        for j, x in enumerate(X):
            min_dist, min_i = None, None
            for i, center in enumerate(self.centroids):
                dist = euclidean(x, center)
                if min_dist is None or min_dist > dist:
                    min_i, min_dist = i, dist
            self.clusters[min_i].add(j)

    def _update_centroids(self, X):
        old_centroids = self.centroids.copy()
        for i, clusters_index in enumerate(self.clusters):
            clusters_x = X[list(clusters_index)]
            self.centroids[i, :] = np.mean(clusters_x, axis=0)
        return old_centroids

    def _is_stop(self, old_centroids, centroids):
        dist = 0
        for clusters1, clusters2 in zip(old_centroids, centroids):
            dist += euclidean(clusters1, clusters2)
        return dist == 0

    def plot(self, preds, X):
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=preds)
