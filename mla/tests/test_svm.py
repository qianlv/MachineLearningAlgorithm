# -*- coding: utf-8 -*-
import numpy as np
from mla.svm import SVM
from mla.base import plot_decision_bounary
from mla.base import Poly
from mla.base import normal_feature
from sklearn.svm import SVC


def test_svm():
    linear = Poly(degree=1)
    poly_3d = Poly(degree=3)
    poly_2d = Poly(degree=2)
    model = SVM(kernerl=poly_3d)
    data = np.loadtxt("../datesets/ex2data1.txt", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    X = normal_feature(X)
    zeros_label = (y == 0)
    y[zeros_label] = -1

    model.train_fit(X, y)
    print((model.predict(X) == y).mean())
    plot_decision_bounary(X, y, model)
