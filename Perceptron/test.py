#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PLA import PLA
from PLA import DualPLA
import numpy as np

if __name__ == '__main__':
    model = PLA(1)
    X = np.array([3, 3, 4, 3, 1, 1])
    X = X.reshape(3, 2)
    Y = np.array([1, 1, -1])
    w, b = model.trainFit(X, Y)
    print(w, b)
    DualModel = DualPLA(1)
    dw, db = DualModel.trainFit(X, Y, 10)
    print(dw, db)
