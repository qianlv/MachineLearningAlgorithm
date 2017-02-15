#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mla.perceptron import PLA
from mla.perceptron import DualPLA
import numpy as np


def test_perceptron():
    X = np.array([3, 3, 4, 3, 1, 1])
    X = X.reshape(3, 2)
    Y = np.array([1, -1, -1])
    model = PLA(1)
    w, b = model.train_fit(X, Y)
    print(w, b)
    Lable = model.predict(X)
    assert (Lable - Y == 0).all()

    Y = np.array([1, -1, -1])
    dualModel = DualPLA(1)
    dw, db = dualModel.train_fit(X, Y)
    print(dw, db)
    Lable = dualModel.predict(X)
    assert (Lable - Y == 0).all()
