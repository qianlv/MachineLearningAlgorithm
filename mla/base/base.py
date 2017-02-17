# -*- coding: utf-8 -*-
import numpy as np


def add_intercept(X, val):
    b = np.zeros([X.shape[0], 1])
    b[:] = val
    return np.concatenate([b, X], axis=1)


def get_split_mask(X, column, value, relation):
    left_mask = relation(X[:, column], value)
    right_mask = ~relation(X[:, column], value)
    return left_mask, right_mask


def split_dataset(X, column, value, relation):
    less_mask, greater_mask = get_split_mask(
            X, column, value, relation[column])
    return X[less_mask], X[greater_mask]


def squared_error(actual, pred):
    return (actual - pred) ** 2


def mean_squared_error(actual):
    mean = np.mean(actual)
    return squared_error(actual, mean)


def pGini(y, lables):
    n_samples = np.shape(y)[0]
    gini = 1.0
    for lable in lables:
        n_lable = np.sum(y == lable)
        gini -= (n_lable / n_samples) ** 2
    return gini


def accuracy(actual, pred):
    return np.mean(actual == pred)
