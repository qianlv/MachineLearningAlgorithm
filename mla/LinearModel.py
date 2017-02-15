#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mla.base import add_intercept


class BasicRegression:

    def __init__(self, eta=0.01, penalty='None', gamma=0.01,
                 tolerance=0.0001, max_iters=10000):
        self._eta = eta
        self._penalty = penalty
        self._gamma = gamma
        self._tolerance = tolerance
        self._max_iters = max_iters
        self.errors = []

    def _setup_input(self, X, y):
        self.X = np.array(X)
        if self.X.ndim == 1:
            self.n_samples, self.n_features = self.X.shape[0], 1
            self.X = self.X.reshape((self.n_samples, self.n_features))
        else:
            self.n_samples, self.n_features = self.X.shape
        self.y = y

    def train_fit(self, X, y, normal=False):
        self._setup_input(X, y)
        if normal:
            self.y = self._featureNormal(self.y)
            self.X = self._featureNormal(self.X)
        self._gradient_descent(self.X, self.y)
        return self.w

    def _cost(self, X, y):
        raise NotImplementedError()

    def loss(self):
        raise NotImplementedError()

    def _featureNormal(self, X):
        self.mu = np.mean(X, 0)
        self.sigma = np.std(X, 0)
        return (X - self.mu) / self.sigma

    def _gradient_descent(self, X, y):
        X = add_intercept(X, 1)
        self.w = np.zeros((1, self.n_features + 1))
        for it in range(self._max_iters):
            cost, grad = self._cost(X, y, self.w)
            print("iter =", it, "w =", self.w, "cost = ", cost, "grad =", grad)
            self.errors.append(cost)
            if cost <= self._tolerance:
                break

            self.w -= self._eta * grad
        self.w = self.w.flatten()

    def predict(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()


class LinearRegression(BasicRegression):

    """ 梯度下降法实现线性回归分类 """

    def _cost(self, X, y, w):
        cost = np.dot(w, X.T) - y
        grad = np.dot(cost, X) / self.n_samples
        cost = np.dot(cost, cost.T) / (2 * self.n_samples)
        return cost, grad

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
        return 1 / (1 + np.exp(-X))

    def _cost(self, X, y, w):
        s = self.sigmod(np.dot(w, X.T))
        #  print ("s shape:", y.shape, s.shape, (y * np.log(s)).shape)
        #  print("s = ", s)
        cost = -np.sum(
                y * np.log(s) + (1 - y) * np.log(1 - s)) / self.n_samples
        grad = np.dot((s - y), X) / self.n_samples
        return cost, grad

    def predict(self, X):
        X = add_intercept(X, 1)
        s = np.dot(self.w, X.T)
        return self.sigmod(np.dot(self.w, X.T))

    def loss(self):
        y = -self.y.reshape((self.y.shape[0], 1))
        s = y * np.dot(self.w, self.X.T)
        cost = np.sum(np.log(1 + np.exp(s)))
        return cost
