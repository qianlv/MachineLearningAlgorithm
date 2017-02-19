# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sp_stats


class RandomForest(object):

    """ RandomForest 算法

    :base_model: 基础训练算法
    :model_params: 基础训练算法的参数
    """

    def __init__(self, base_model, model_params=(),
                 bootstrap_size=None, max_features=None):
        """
        _model_data_mask: 记录每次迭代使用的数据下表
        """
        self._base_model = base_model
        self._model_params = model_params
        self._trained_models = None
        self._X = None
        self._y = None
        self._model_data_mask = None
        self._boostrap_size = bootstrap_size
        self._max_features = max_features

    def train_fit(self, X, y, max_iters=5):
        n_samples, n_features = np.shape(X)
        self._trained_models = []
        self._model_data_mask = []
        self._X = X
        self._y = y
        if self._boostrap_size is None:
            self._boostrap_size = n_samples
        for it in range(max_iters):
            sample_mask = self._random_data(n_samples)
            train_X = X[sample_mask]
            train_y = y[sample_mask]
            model = self._base_model(*self._model_params)
            model.train_fit(train_X, train_y, max_features=self._max_features)
            self._trained_models.append(model)
            self._model_data_mask.append(set(sample_mask))

    def _random_data(self, n_samples):
        assert self._boostrap_size
        mask = np.array(
            [int(np.random.uniform(0, n_samples))
             for _ in range(self._boostrap_size)])
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
