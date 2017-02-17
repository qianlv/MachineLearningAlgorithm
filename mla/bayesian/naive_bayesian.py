# -*- coding: utf-8 -*-
import numpy as np
from mla.base import get_split_mask_by_eq
from mla.base import squared_error


class NaiveBaysian:

    def __init__(self):
        self._labels = None
        self._prior_probs = None
        self._condition_probs = {}
        self._is_discrete_features = None
        self._feature_labels = None

    def _cal_prior_prob(self, y):
        n_samples = np.shape(y)[0]
        self._prior_probs = {}
        n_labels = np.shape(self._labels)[0]
        for label in self._labels:
            n_label_samples = (np.sum(y == label) + 1)
            self._prior_probs[label] = n_label_samples / (n_samples + n_labels)

    def gauss(self, mean, var, x):
        factor = 1.0 / np.sqrt(2 * np.pi * var)
        tmp = np.exp(-(x - mean) ** 2 / (2 * var))
        return factor * tmp

    def _cal_mean_and_var(self, X):
        return np.mean(X), np.var(X, ddof=1)

    def _cal_condition_prob(self, X, y):
        n_samples, n_features = np.shape(X)
        for label in self._labels:
            eq_y_mask = (y == label)
            n_label_samples = np.sum(eq_y_mask)
            self._condition_probs[label] = {}
            for f in range(n_features):
                self._condition_probs[label][f] = {}
                if self._is_discrete_features[f]:
                    n_feature_labels = len(self._feature_labels[f])
                    for f_label in self._feature_labels[f]:
                        self._condition_probs[label][f][f_label] =      \
                            (np.sum(X[eq_y_mask, f] == f_label) + 1) /  \
                            (n_label_samples + n_feature_labels)
                else:
                    self._condition_probs[label][f] = \
                            self._cal_mean_and_var(X[eq_y_mask, f])

    def train_fit(self, X, y, discrete_features=None):
        if discrete_features:
            self._is_discrete_features = discrete_features
        else:
            self._is_discrete_features = [True for _ in range(np.shape(X)[1])]
        np_unique = np.lib.arraysetops.unique
        self._labels = np_unique(y)
        n_samples, n_features = np.shape(X)
        self._feature_labels = {}
        for f in range(n_features):
            self._feature_labels[f] = np_unique(X[:, f])

        self._cal_prior_prob(y)
        self._cal_condition_prob(X, y)

    def predict(self, X):
        n_samples, n_features = np.shape(X)
        y = []
        for row_x in X:
            max_prob = None
            max_label = None
            for label in self._labels:
                prob = np.log(self._prior_probs[label])
                for f in range(n_features):
                    if self._is_discrete_features[f]:
                        prob *= np.log(
                            self._condition_probs[label][f][row_x[f]])
                    else:
                        prob *= np.log(self.gauss(
                                       *self._condition_probs[label][f],
                                       row_x[f]
                                       ))
                print(label, prob)
                if not max_prob or max_prob < prob:
                    max_prob = prob
                    max_label = label
            y.append(max_label)
        return y
