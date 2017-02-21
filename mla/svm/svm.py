# -*- coding: utf-8 -*-
import numpy as np
from mla.base import Poly
Linear = Poly(degree=1)


class SVM(object):
    """ simple SVO """
    def __init__(self, C=1, tol=1e-3, kernerl=Linear):
        self.C = C
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.kernerl = Linear
        self.K = None
        self.svm_idx = None
        self.X = None
        self.y = None
        self.E = None

    def train_fit(self, X, y, max_iters=100):
        self.X = np.atleast_2d(X)
        self.y = y
        n_samples, n_features = np.shape(self.X)

        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i, j] = self.kernerl(self.X[i, :], self.X[j, :])

        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.svm_idx = np.array(range(0, n_samples))
        self.E = np.zeros(n_samples)

        it = 0
        while it < max_iters:
            self.updateE()
            first = self._find_first_alpha(n_samples)
            num = 0
            #  for first in range(n_samples):
            second = self._find_second_alpha(first, n_samples)
                #  second = self.random_index(first, n_samples)
            num += self._single_fit(first, second)
            if num == 0:
                it += 1

        self.svm_idx = np.where(self.alpha > 0)[0]
        print("alpha: ", self.alpha, self.svm_idx)

    def updateE(self):
        for i, x in enumerate(self.X):
            self.E[i] = self.predict_row(x) - self.y[i]

    def _single_fit(self, i, j):
        e_i = self.E[i]

        eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
        if eta >= 0:
            return False

        e_j = self.E[j]

        L, H = self._find_bound(i, j)
        if L == H:
            return False

        alpha_oi, alpha_oj = self.alpha[i], self.alpha[j]
        self.alpha[j] -= self.y[j] * (e_i - e_j) / eta
        self.alpha[j] = self._clip(self.alpha[j], L, H)
        diff_alpha_j = self.alpha[j] - alpha_oj
        if abs(diff_alpha_j) < 1e-5:
            return False

        self.alpha[i] += self.y[i] * self.y[j] * \
            (alpha_oj - self.alpha[j])
        diff_alpha_i = self.alpha[i] - alpha_oi

        b_i = self.b - e_i \
            - self.y[i] * self.K[i, i] * diff_alpha_i \
            - self.y[j] * self.K[i, j] * diff_alpha_j
        b_j = self.b - e_j \
            - self.y[i] * self.K[i, j] * diff_alpha_i \
            - self.y[j] * self.K[j, j] * diff_alpha_j

        if 0 < self.alpha[i] < self.C:
            self.b = b_i
        elif 0 < self.alpha[j] < self.C:
            self.b = b_j
        else:
            self.b = (b_i + b_j) * 0.5
        return True

    def _find_first_alpha(self, n_samples):
        max_err = 0.0
        first = None
        for i in range(n_samples):
            err = self.E[i]
            if not self._is_kkt_point(i):
                if abs(max_err) < abs(err):
                    max_err = err
                    first = i
        return first

    def _find_second_alpha(self, first, n_samples):
        max_err = 0.0
        second = None
        for i in range(n_samples):
            if i == first:
                continue
            if abs(self.E[i] - self.E[first]) > max_err:
                max_err = abs(self.E[i] - self.E[first])
                second = i
        return second

    def _is_kkt_point(self, i):
        err = self.E[i]
        if 0 < self.alpha[i] < self.C:
            if abs(self.y[i] * err) < self.tol:
                return True
        elif self.alpha[i] == 0:
            if self.y[i] * err > -self.tol:
                return True
        elif self.alpha[i] == self.C:
            if self.y[i] * err < self.tol:
                return True
        return False

    def _find_bound(self, i, j):
        if self.y[i] == self.y[j]:
            return max(0, self.alpha[j] + self.alpha[i] - self.C), \
                   min(self.C, self.alpha[i] + self.alpha[j])
        else:
            return max(0, self.alpha[j] - self.alpha[i]), \
                   min(self.C, self.C + self.alpha[j] - self.alpha[i])

    def _clip(self, alpha, L, H):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def predict(self, X):
        pred_y = np.array([np.sign(self.predict_row(x)) for x in X])
        return pred_y

    def predict_row(self, x):
        kernerl_value = self.kernerl(self.X[self.svm_idx], x)
        scores = np.dot(
            self.alpha[self.svm_idx] * self.y[self.svm_idx],
            kernerl_value.T) + self.b
        return scores

    def random_index(self, i, n_samples):
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
