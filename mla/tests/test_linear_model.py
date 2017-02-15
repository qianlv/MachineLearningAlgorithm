# -*- coding: utf-8 -*-
from mla.LinearModel import LinearRegression
from mla.LinearModel import LogisticRegression
from numpy import loadtxt


def test_linear_model():
    linear = LinearRegression(max_iters=1500)
    data = loadtxt("../datesets/ex1data1.txt", delimiter=",")
    X = data[:, 0]
    Y = data[:, 1]
    #  print(X, Y)
    linear.train_fit(X, Y, normal=True)
    linear.plot()
