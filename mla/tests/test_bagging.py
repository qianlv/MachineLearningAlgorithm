# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mla.ensemble import Bagging
from mla.ensemble import DecisionTreeClassifier
from mla.base import normal_feature


def test_bagging():
    #  data = np.loadtxt("../datesets/ex2data1.txt", delimiter=",")
    iris = datasets.load_iris()
    model = Bagging(DecisionTreeClassifier, model_params=(None, 1, 4))
    X = iris.data[:, [0, 2]]
    y = iris.target
    X = normal_feature(X)
    model.train_fit(X, y, max_iters=5)

    #  X = np.array(sorted(X.flatten()))
    #  X = X.reshape((X.shape[0], 2))
    #  plt.plot(X, model.predict(X))
    h = 0.1  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()
