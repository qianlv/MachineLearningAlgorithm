# -*- coding: utf-8 -*-
import numpy as np
import operator


class DecisionStump(object):

    """ decision stump """

    def __init__(self):
        self._stump = None
        self._op = None
        self._weight = None
        self._split_feature = None

    def _get_split(self, X, y, steps=15):
        bottom = np.min(X)
        up = np.max(X)
        steps = (up - bottom) / steps
        n_samples = np.shape(y)[0]
        min_err, min_stump, min_op = None, None, None

        for stump in np.arange(bottom, up, steps):
            for op in (operator.lt, operator.ge):
                result = np.ones(n_samples)
                result[op(X, stump)] = -1
                error = np.dot(~(y == result), self._weight)
                if min_err is None or min_err > error:
                    min_err = error
                    min_stump = stump
                    min_op = op
        return min_err, min_stump, min_op

    def train_fit(self, X, y, w):
        n_samples, n_features = np.shape(X)
        min_err, min_stump, min_op, min_split_feature = None, None, None, None
        self._weight = w

        for f in range(n_features):
            error, stump, op = self._get_split(X[:, f], y)
            if min_err is None or min_err > error:
                min_err = error
                min_stump = stump
                min_op = op
                min_split_feature = f

        self._stump = min_stump
        self._op = min_op
        self._split_feature = min_split_feature
        return min_err

    def predict(self, X):
        n_samples = np.shape(X)[0]
        result = np.ones(n_samples)
        result[self._op(X[:, self._split_feature], self._stump)] = -1
        return result
