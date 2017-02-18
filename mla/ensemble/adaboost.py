# -*- coding: utf-8 -*-
import numpy as np
from mla.ensemble import DecisionStump


class AdaBoost(object):

    def __init__(self, weeker_model=DecisionStump):
        self._weeker_classifys = None
        self._alphas = None
        self._weeker_model = weeker_model
        pass

    def train_fit(self, X, y, max_iters=4):
        self._weeker_classifys = []
        self._alphas = np.zeros(max_iters)

        n_samples, n_features = np.shape(X)
        weights = np.array([1 / n_samples for _ in range(n_samples)])
        for it in range(max_iters):
            model = self._weeker_model()
            error = model.train_fit(X, y, weights)
            self._weeker_classifys.append(model)

            alpha = 0.5 * np.log((1.0 - error) / error)
            self._alphas[it] = alpha

            pred = np.exp(-model.predict(X) * y * alpha)
            z = np.sum(weights * pred)
            weights = (weights / z) * pred 
        print("alpha: ", self._alphas)

    def predict(self, X):
        n_samples, n_features = np.shape(X)
        pred = np.zeros(n_samples)
        for alpha, weeker in zip(self._alphas, self._weeker_classifys):
            pred += alpha * weeker.predict(X)
        return np.sign(pred)
