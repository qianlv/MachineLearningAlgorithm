# -*- coding: utf-8 -*-
from mla.clusters import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


def test_kmeans():
    X, y = make_blobs(
        n_samples=2000,
        n_features=2,
        centers=[[-1, 1], [-1, 3], [-2, 2]],
        cluster_std=[0.4, 0.5, 0.2],)
    model = KMeans(k=3)
    plt.figure()
    plt.subplot(121)
    preds = model.predict(X, max_iters=1000)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=preds)
    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()
