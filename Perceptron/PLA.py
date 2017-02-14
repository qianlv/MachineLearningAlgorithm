#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class PLA:
    """
    感知机算法,

    Model: f(x) = sign(w * x + b)
    self._X : [[x1], [x2], ... x[m]]
    self._Y : [y1, y2, ..., ym]

    """
    def __init__(self, X, Y, eta):
        self._eta = eta
        self._X = np.array(X)
        self._Y = np.array(Y).transpose()
        self._SampleNum = self._X.shape[0]
        self._SampleDem = self._X.shape[1]
        self._w = np.zeros((1, self._SampleDem))
        self._b = 0

    def trainFit(self, maxStep=None):
        self.stochasticGradientDescent(maxStep)
        return self._w, self._b

    def classify(self, i, w, b):
        return self._Y[i] * (np.dot(w, self._X[i, :]) + b) <= 0

    def update(self, i):
        self._w = self._w + self._eta * self._Y[i] * self._X[i]
        self._b = self._b + self._eta * self._Y[i]

    def stochasticGradientDescent(self, maxStep=None):
        step = 0
        while not maxStep or step < maxStep:
            step += 1
            for i in range(self._SampleNum):
                if self.classify(i, self._w, self._b):
                    self.update(i)
                    break
            else:
                break
        self.w = np.concatenate(
                    [np.array(self._b).reshape((1, 1)), self._w],
                    axis=1)

    def _add_intercept(self, X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def predict(self, X):
        return np.sign(np.dot(self._add_intercept(X), self.w.T)).flatten()


class DualPLA:

    """ 原始 PLA 的对偶形式

    Model: f(x) = sign(w * x + b)
    self._X : [[x1], [x2], ... x[m]]
    self._Y : [y1, y2, ..., ym]
    """

    def __init__(self, X, Y, eta):
        self._eta = eta
        self._X = np.array(X)
        self._Y = np.array(Y).transpose()
        self._SampleNum = self._X.shape[0]
        self._SampleDem = self._X.shape[1]
        self._w = np.zeros((1, self._SampleDem))
        self._alpha = np.zeros((1, self._SampleNum))
        self._gram = np.dot(self._X, self._X.T)
        self._b = 0

    def classify(self, i):
        scores = np.sum(np.dot((self._alpha * self._Y), self._gram[i, :]))
        scores = (scores + self._b) * self._Y[i]
        return scores <= 0

    def update(self, i):
        self._alpha[:, i] = self._alpha[:, i] + self._eta
        self._b = self._b + self._eta * self._Y[i]

    def trainFit(self, maxStep=None):
        step = 0
        while not maxStep or step < maxStep:
            step += 1
            for i in range(self._SampleNum):
                if self.classify(i):
                    self.update(i)
                    print("Iteration :", step, "misclassfied x is ", (i + 1),
                          "alpha: ", self._alpha, "b: ", self._b)
                    break
            else:
                break
        self._w = np.dot(self._alpha * self._Y, self._X)
        self.w = np.concatenate(
                    [np.array(self._b).reshape((1, 1)), self._w],
                    axis=1)
        return self._w, self._b

    def _add_intercept(self, X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def predict(self, X):
        return np.sign(np.dot(self._add_intercept(X), self.w.T)).flatten()
