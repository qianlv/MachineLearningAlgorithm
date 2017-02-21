# -*- coding: utf-8 -*-
import numpy as np


class Poly(object):
    def __init__(self, degree=1, alpha=0, beta=1):
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x, y):
        return (self.alpha + self.beta * np.dot(x, y.T)) ** self.degree

    def __repr__(self):
        return "{0} degrees Poly kernel, params: ({1}, {2})".format(
                self.degree, self.alpha, self.beta)


class RBF(object):
    def __init__(self, beta=1):
        if beta <= 0:
            raise ValueError("RBF beta must beta {0} > 0".format(beta))
        self.beta = beta

    def __call__(self, x, y):
        dist = np.dot(x - y, x - y)
        return np.exp(-self.beta * dist)

    def __repr__(self):
        return "RBF kernerl"
