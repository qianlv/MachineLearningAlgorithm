# -*- coding: utf-8 -*-
import numpy as np


def add_intercept(X, val):
    b = np.zeros([X.shape[0], 1])
    b[:] = val
    return np.concatenate([b, X], axis=1)
