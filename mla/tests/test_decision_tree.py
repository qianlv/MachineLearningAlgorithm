# -*- coding: utf-8 -*-
from mla.ensemble import DecisionTree
from mla.base import accuracy
import numpy as np
import matplotlib.pyplot as plt


def test_decision_tree_classify():
    X = np.array([
        ['teenager',        'no',   'no',   0.0],
        ['teenager',        'no',   'no',   1.0],
        ['teenager',        'yes',  'no',   1.0],
        ['teenager',        'yes',  'yes',  0.0],
        ['teenager',        'no',   'no',   0.0],
        ['senior citizen',  'no',   'no',   0.0],
        ['senior citizen',  'no',   'no',   1.0],
        ['senior citizen',  'yes',  'yes',  1.0],
        ['senior citizen',  'no',   'yes',  2.0],
        ['senior citizen',  'no',   'yes',  2.0],
        ['old pepple',      'no',   'yes',  2.0],
        ['old pepple',      'no',   'yes',  1.0],
        ['old pepple',      'yes',  'no',   1.0],
        ['old pepple',      'yes',  'no',   2.0],
        ['old pepple',      'no',   'no',   0.0],
        ])

    y = np.array([-1, -1, +1, +1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1])

    model = DecisionTree(tree_type="clf")
    model.train_fit(X, y)
    print(model.predict(X) == y)


def test_decision_tree_regresson():
    data = np.loadtxt("../datesets/decision_tree/exp.txt", delimiter="\t")
    test_data = np.loadtxt(
        "../datesets/decision_tree/expTest.txt", delimiter="\t")
    X = data[:, :-1]
    y = data[:, -1]
    model = DecisionTree(tree_type="reg", tol_err=1, tol_nset=4)
    model.train_fit(X, y)
    print("error rate before prune: ", model.loss(test_data))
    model.prune(test_data)
    print("error rate after prune: ", model.loss(test_data))
    plt.scatter(X, y, c='r')
    X = np.array(sorted(X.flatten()))
    X = X.reshape((X.shape[0], 1))
    plt.plot(X, model.predict(X))
    plt.show()
