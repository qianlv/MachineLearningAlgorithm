#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
from mla.base import add_intercept


class BasicRegression:

    def __init__(self, method="sgd", eta=0.01, penalty='None', gamma=0.01,
                 tolerance=0.0001, max_iters=10000):
        self._eta = eta
        self._penalty = penalty
        self._gamma = gamma
        self._tolerance = tolerance
        self._max_iters = max_iters
        self.errors = []
        if method == "sgd":
            self._cost = self._costStocGrad
        else:
            self._cost = self._costGrad

    def _setup_input(self, X, y):
        self.X = np.array(X)
        if self.X.ndim == 1:
            self.n_samples, self.n_features = self.X.shape[0], 1
            self.X = self.X.reshape((self.n_samples, self.n_features))
        else:
            self.n_samples, self.n_features = self.X.shape
        self.y = y

    def train_fit(self, X, y):
        self._setup_input(X, y)
        self._gradient_descent(self.X, self.y)
        return self.w

    def _costGrad(self, X, y, w):
        raise NotImplementedError()

    def _costStocGrad(self, X, y, w):
        raise NotImplementedError()

    def loss(self):
        raise NotImplementedError()

    def _gradient_descent(self, X, y):
        X = add_intercept(X, 1)
        self.w = np.ones((1, self.n_features + 1))
        for it in range(self._max_iters):
            grad = self._cost(X, y, self.w)
            self.w -= (self._eta + 1 / (2 + it)) * grad
            #  print("iter =", it, "w =", self.w, "grad =", grad)
        self.w = self.w.flatten()

    def predict(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()


class LinearRegression(BasicRegression):

    """ 梯度下降法实现线性回归分类 """

    def _costGrad(self, X, y, w):
        cost = np.dot(w, X.T) - y
        grad = np.dot(cost, X) / self.n_samples
        return grad

    def _costStocGrad(self, X, y, w):
        #  randindex = sorted(random.sample(range(self.n_samples), 3))
        randindex = int(random.uniform(0, self.n_samples))
        return self._costGrad(X[randindex], y[randindex], w)

    def loss(self):
        cost = np.dot(self.w, self.X.T) - self.y
        cost = np.dot(cost, cost.T) / (2 * self.n_samples)
        return cost

    def predict(self):
        return sp.poly1d(self.w[::-1])

    def plot(self):
        plt.scatter(self.X, self.y, marker='x', c='r')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis([np.min(self.X), np.max(self.X),
                  np.min(self.y), np.max(self.y)])
        mx = np.linspace(np.min(self.X), np.max(self.X), 100)
        pred = self.predict()
        plt.plot(mx, pred(mx), label="predict")
        plt.legend()
        plt.autoscale(tight=True)
        plt.show()


class LogisticRegression(BasicRegression):

    """Docstring for LogisticRegression. """

    def sigmod(self, X):
        return 1.0 / (1 + np.exp(-X))

    def _costGrad(self, X, y, w):
        s = self.sigmod(np.dot(w, X.T))
        grad = np.dot((s - y), X)
        return grad

    def _costStocGrad(self, X, y, w):
        #  randindex = sorted(sample(range(self.n_samples), 3))
        randindex = int(random.uniform(0, self.n_samples))
        return self._costGrad(X[randindex].reshape((1, self.n_features + 1)),
                              y[randindex], w)

    def predict(self, X):
        X = add_intercept(X, 1)
        s = np.dot(self.w, X.T)
        return self.sigmod(s)

    def error_rate(self, X, y):
        pred = (self.predict(X) > 0.5)
        pred = np.array(pred, dtype=float)
        error = (pred != y).mean()
        return error
