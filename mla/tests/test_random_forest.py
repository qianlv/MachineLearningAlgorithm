# -*- coding: utf-8 -*-
from sklearn import datasets
from mla.ensemble import RandomForest
from mla.ensemble import DecisionTreeClassifier
from mla.base import plot_decision_bounary


def test_random_forest():
    iris = datasets.load_iris()
    model = RandomForest(
        DecisionTreeClassifier, model_params=(None, 1, 4), max_features=1)
    X = iris.data[:, [0, 2]]
    y = iris.target
    model.train_fit(X, y, max_iters=100)
    plot_decision_bounary(X, y, model)
