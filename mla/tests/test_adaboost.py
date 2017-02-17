# -*- coding: utf-8 -*-
import numpy as np
from mla.ensemble import AdaBoost


def test_adaboost():
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X = X.reshape((10, 1))
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    model = AdaBoost()
    model.train_fit(X, y, max_iters=3)
    print(model.predict(X))
