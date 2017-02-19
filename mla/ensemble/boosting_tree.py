# -*- coding: utf-8 -*-
import numpy as np
from mla.ensemble import DecisionTreeRegressor


class BoostingTree(object):

    def __init__(self,
                 base_model=DecisionTreeRegressor,
                 model_params=(2,)):
        self._base_model = base_model
        self._model_params = model_params
        self._trained_models = None

    def train_fit(self, X, y, max_iters=5):
        self._trained_models = []
        residual = y
        for it in range(max_iters):
            model = self._base_model(*self._model_params)
            model.train_fit(X, residual)
            self._trained_models.append(model)
            pred = model.predict(X)
            residual -= pred
            print("residual: ", residual)

    def predict(self, X):
        y = np.zeros(np.shape(X)[0])
        for model in self._trained_models:
            y += model.predict(X)
        return y

