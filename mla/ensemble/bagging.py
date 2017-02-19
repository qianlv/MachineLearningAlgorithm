# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sp_stats
from mla.perceptron import PLA


class Bagging(object):

    """ Bagging 算法

    :base_model: 基础训练算法
    :model_params: 基础训练算法的参数
    """

    def __init__(self, base_model, model_params=()):
        """
        _model_data_mask: 记录每次迭代使用的数据下表
        """
        self._base_model = base_model
        self._model_params = model_params
        self._trained_models = None
        self._X = None
        self._y = None
        self._model_data_mask = None

    def train_fit(self, X, y, max_iters=5):
        n_samples, _ = np.shape(X)
        self._trained_models = []
        self._model_data_mask = []
        self._X = X
        self._y = y
        for it in range(max_iters):
            mask = self._random_data(n_samples, n_samples)
            model = self._base_model(*self._model_params)
            model.train_fit(X[mask], y[mask])
            no_train_mask = np.array([True] * n_samples)
            no_train_mask[np.unique(mask)] = False
            test_n_samples, _ = X[no_train_mask].shape
            model.prune(np.concatenate([X[no_train_mask], y[no_train_mask].reshape((test_n_samples, 1))], axis=1))
            self._trained_models.append(model)
            self._model_data_mask.append(set(mask))

    def _random_data(self, n_samples, k):
        mask = np.array(
            [int(np.random.uniform(0, n_samples)) for _ in range(k)])
        return mask

    def predict(self, X):
        n_samples, _ = np.shape(X)
        Y = np.zeros((n_samples, len(self._trained_models)))
        for i, model in enumerate(self._trained_models):
            Y[:, i] = model.predict(X)
        y = sp_stats.mode(Y, axis=1)[0].flatten()
        return y

    def out_of_bag_estimate(self):
        """ 包外估计(泛化性能)
        """
        n_samples, _ = np.shape(self._X)
        sum_error = 0.0
        for j, model in enumerate(self._trained_models):
            mask = self._model_data_mask[j]
            no_train_mask = np.array([True] * n_samples)[mask] = False
            pred = self.predict(self._X[no_train_mask])
            sum_error += np.sum(pred != self._y[no_train_mask])
        return sum_error / n_samples
