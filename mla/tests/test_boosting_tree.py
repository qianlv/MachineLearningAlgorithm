# -*- coding: utf-8 -*-
import numpy as np
from mla.ensemble import BoostingTree


def test_boosting_tree():
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X = X.reshape((10, 1))
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    model = BoostingTree()
    model.train_fit(X, y)
