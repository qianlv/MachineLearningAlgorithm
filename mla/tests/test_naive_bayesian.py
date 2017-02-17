# -*- coding: utf-8 -*-

import numpy as np
from mla.bayesian import NaiveBaysian


def test_naiva_bayesia_continuous_varibale():
    X = np.array([
        [6.0,   180,    12],
        [5.92,  190,    11],
        [5.58,  170,    12],
        [5.92,  165,    10],
        [5.0,   100,    6],
        [5.5,   150,    8],
        [5.42,  130,    7],
        [5.75,  150,    9],
        ])

    y = np.array([
        'male',
        'male',
        'male',
        'male',
        'female',
        'female',
        'female',
        'female',
        ])

    model = NaiveBaysian()
    model.train_fit(X, y, [False, False, False])
    test_point = np.array([[6, 130, 8]])
    print(model.predict(test_point))

def test_naiva_bayesia_discrete():
    X = np.array([
        [1, 'S'],
        [1, 'M'],
        [1, 'M'],
        [1, 'S'],
        [1, 'S'],
        [2, 'S'],
        [2, 'M'],
        [2, 'M'],
        [2, 'L'],
        [2, 'L'],
        [3, 'L'],
        [3, 'M'],
        [3, 'M'],
        [3, 'L'],
        [3, 'L'],
    ])

    y = np.array([
        -1,
        -1,
        +1,
        +1,
        -1,
        -1,
        -1,
        +1,
        +1,
        +1,
        +1,
        +1,
        +1,
        +1,
        -1,
    ])

    model = NaiveBaysian()
    model.train_fit(X, y)
    print(model.predict([['2', 'S'], ['2', 'L']]))
