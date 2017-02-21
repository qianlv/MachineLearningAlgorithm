# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean


class KNN(object):

    def __init__(self, k, distance=euclidean):
        self.k = k
        self.X = None
        self.y = None
        self.distance = distance

    def train_fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        X = np.atleast_2d(X)
        return [self._predict_row_x(x) for x in X]

    def _predict_row_x(self, x):
        distances = (self.distance(x, row_x) for row_x in self.X)
        neighbors = sorted(((dist, i) for i, dist in enumerate(distances)),
                           key=lambda x: x[0])
        nearest_neighbors = [self.y[i] for _, i in neighbors[:self.k]]
        return self.voting(nearest_neighbors)

    def voting(self, nearest_neighbors):
        raise NotImplementedError()


class KNNClassifier(KNN):
    def voting(self, nearest_neighbors):
        most_labels = Counter(nearest_neighbors).most_common(1)[0][0]
        return most_labels


class KNNRegresser(KNN):
    def voting(self, nearest_neighbors):
        return np.mean(nearest_neighbors)
