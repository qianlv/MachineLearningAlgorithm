#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class PLA:
    def __init__(self, eta):
        self._eta = eta

    def trainFit(self, Input, Output, maxStep=None):
        self._w, self._b = PLA.stochasticGradientDescent(
                        Input, Output, self._eta, maxStep)
        return self._w, self._b

    def predict(self, x):
        return np.sign(np.dot(self._w.T, x) + self._b)

    @staticmethod
    def stochasticGradientDescent(Input, Output, eta, maxStep=None):
        m = Input.shape[0]
        n = Input.shape[1]
        w = np.zeros((1, n))
        b = 0
        step = 0
        while not maxStep or step < maxStep:
            step += 1
            for i in range(m):
                if Output[i] * (np.dot(Input[i], w.T) + b) <= 0:
                    w = w + eta * Output[i] * Input[i]
                    b = b + eta * Output[i]
                    print(w, b)
                    break
            else:
                return w, b
        return w, b


class DualPLA:

    """ 原始 PLA 的对偶形式"""

    def __init__(self, eta):
        self._eta = eta

    def trainFit(self, Input, Output, maxStep=None):
        gram = np.dot(Input, Input.T)
        m = Input.shape[0]
        self._alpha = np.zeros((1, m))
        self._b = 0

        step = 0
        while not maxStep or step < maxStep:
            step += 1
            for i in range(m):
                scores = np.sum(np.dot((self._alpha * Output), gram[i, :]))
                scores = (scores + self._b) * Output[i]
                if scores <= 0:
                    self._alpha[:, i] = self._alpha[:, i] + self._eta
                    self._b = self._b + self._eta * Output[i]
                    break
            else:
                break
        self._w = np.dot(self._alpha * Output, Input)
        return self._w, self._b

    def predict(self, x):
        return np.sign(np.dot(self._w.T, x) + self._b)
