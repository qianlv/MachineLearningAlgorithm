# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from mla.ensemble import AdaBoost
from mla.base import plot_decision_bounary


def test_adaboost():
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X = X.reshape((10, 1))
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    model = AdaBoost()
    model.train_fit(X, y, max_iters=3)
    assert (model.predict(X) - y == 0).all()
    print(model.predict(X))
