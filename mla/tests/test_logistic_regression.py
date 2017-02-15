# -*- coding: utf-8 -*-
from mla.LinearModel import LogisticRegression
import numpy as np


def test_logistic_regress():
    model = LogisticRegression(tolerance=0.204, eta=0.0001, max_iters=10000000)
    data = np.loadtxt("../datesets/ex2data1.txt", delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1]
    w = model.train_fit(X, Y)
    print(model.predict(np.array([45, 85]).reshape((1, 2))))
