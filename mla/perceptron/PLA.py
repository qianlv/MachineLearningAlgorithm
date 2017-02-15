#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from mla.base import add_intercept


class PLA:
    """
    感知机算法,

    Model: f(x) = sign(w * x + b)
    self._X : [[x1], [x2], ... x[m]]
    self._Y : [y1, y2, ..., ym]

    """
    def __init__(self, eta, max_iters=None):
        self._eta = eta
        self._max_iters = max_iters

    def train_fit(self, X, y):
        self._stochastic_gradient_descent(X, y)
        return self.w[0, 1:], self.w[0, 0]

    def _stochastic_gradient_descent(self, X, y):
        step = 0
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        self.w = np.zeros((1, self.n_features))
        self.w = add_intercept(self.w, 0)
        X = add_intercept(X, 1)

        while not self._max_iters or step < self._max_iters:
            step += 1
            update_pos = self.classify(X, y)
            if update_pos >= 0:
                self.update(X, y, update_pos)
                print("Itetator {0}, Update Pos: {1}, w: {2}"
                      .format(step, update_pos, self.w))
            else:
                break
        return self.w

    def classify(self, X, y):
        for i in range(self.n_samples):
            if y[i] * np.dot(self.w, X[i, :]) <= 0:
                return i
        return -1

    def update(self, X, y, update_pos):
        self.w += self._eta * y[update_pos] * X[update_pos, :]

    def predict(self, X):
        return np.sign(np.dot(add_intercept(X, 1), self.w.T)).flatten()


class DualPLA:

    """ 原始 PLA 的对偶形式

    Model: f(x) = sign(w * x + b)
    self._X : [[x1], [x2], ... x[m]]
    self._Y : [y1, y2, ..., ym]
    """

    def __init__(self, eta, max_iters=None):
        self._eta = eta
        self._max_iters = max_iters

    def classify(self, y, i):
        scores = np.sum(np.dot((self._alpha * y), self._gram[i, :]))
        scores = (scores + self._b) * y[i]
        return scores <= 0

    def update(self, y, i):
        self._alpha[:, i] = self._alpha[:, i] + self._eta
        self._b = self._b + self._eta * y[i]

    def train_fit(self, X, y):
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        self._w = np.zeros((1, self.n_features))
        self._alpha = np.zeros((1, self.n_samples))

        self._b = 0
        self._gram = np.dot(X, X.T)
        step = 0
        while not self._max_iters or step < self._max_iters:
            step += 1
            for i in range(self.n_samples):
                if self.classify(y, i):
                    self.update(y, i)
                    print("Iteration :", step, "misclassfied x is ", (i + 1),
                          "alpha: ", self._alpha, "b: ", self._b)
                    break
            else:
                break
        self._w = np.dot(self._alpha * y, X)
        self.w = np.concatenate(
                    [np.array(self._b).reshape((1, 1)), self._w],
                    axis=1)
        return self._w, self._b

    def predict(self, X):
        return np.sign(np.dot(add_intercept(X, 1), self.w.T)).flatten()
