# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from mla.LinearModel import LogisticRegression
from mla.base import normal_feature


def plot_fit(X, y, w):
    oneLabel = (y == 1)
    plt.scatter(X[oneLabel, 0], X[oneLabel, 1], s=30, c='red', marker='s')
    plt.scatter(X[~oneLabel, 0], X[~oneLabel, 1], s=30, c='green')

    mx = np.linspace(np.min(X[oneLabel]) - 0.5,
                     np.max(X[oneLabel]) + 0.5, 2000)
    my = (-w[0] - w[1] * mx) / w[2]
    plt.plot(mx, my)
    plt.show()


def test_logistic_regress():
    model = LogisticRegression(eta=0.01, max_iters=50, method="sgd")
    data = np.loadtxt("../datesets/ex2data1.txt", delimiter=",")
    #  data = np.loadtxt("../datesets/testSet.txt", delimiter="\t")
    X = data[:, :-1]
    Y = data[:, -1]
    X = normal_feature(X)
    w = model.train_fit(X, Y)
    print(w)
    plot_fit(X, Y, w)


def test_logistic_regress_test():
    data = np.loadtxt("../datesets/horseColicTraining.txt", delimiter="\t")
    test_data = np.loadtxt("../datesets/horseColicTest.txt", delimiter="\t")
    X = data[:, :-1]
    Y = data[:, -1]
    X = normal_feature(X)
    model = LogisticRegression(eta=0.008, max_iters=1000, method="sgd")
    sum_error = 0.0
    for te in range(10):
        w = model.train_fit(X, Y)
        print(w)
        error = model.error_rate(
                    normal_feature(test_data[:, :-1]),
                    test_data[:, -1])
        print("the error: %f" % error)
        sum_error += error
    print("the sum_error: $f", sum_error / 10.0)
