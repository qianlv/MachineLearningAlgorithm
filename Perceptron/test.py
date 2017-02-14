#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PLA import PLA
from PLA import DualPLA
import numpy as np

if __name__ == '__main__':
    X = np.array([3, 3, 4, 3, 1, 1])
    X = X.reshape(3, 2)
    Y = np.array([1, 1, -1])
    model = PLA(X, Y, 1)
    w, b = model.trainFit()
    print(w, b)
    Lable = model.predict(X)
    print(Lable)

    dualModel = DualPLA(X, Y, 1)
    dw, db = dualModel.trainFit(10)
    print(dw, db)
    Lable = dualModel.predict(X)
    print(Lable)
