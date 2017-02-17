# -*- coding: utf-8 -*-
import numpy as np
import operator
from collections import Counter
from mla.base import get_split_mask
from mla.base import split_dataset
from mla.base import mean_squared_error
from mla.base import squared_error
from mla.base import pGini


class Tree:

    def __init__(self):
        self.is_leaf = False
        self.node_val = None
        self.split_feature = None
        self.counter_samples = None
        self.left = None
        self.right = None

    def query(self, x_row, relation):
        if self.is_leaf:
            return self.node_val
        if relation[self.split_feature](
           x_row[self.split_feature], self.node_val):
            return self.left.query(x_row, relation)
        else:
            return self.right.query(x_row, relation)


"""
TODO:
  1. 预剪枝
  2. 缺失值处理
  3. 多变量决策
"""


class DecisionTree(object):

    """ CART 算法生成决策树 """

    Operators = {
        '==': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
    }

    def __init__(self, tree_type="reg", limit_depth=None,
                 tol_err=0, tol_nset=1):
        if tree_type == "reg":
            self._calculate_error = self._calculate_least_squares_error
        elif tree_type == "clf":
            self._calculate_error = self._calculate_gini
        elif tree_type == "model":
            self._calculate_error = self._calculate_linear_model_error
        else:
            raise ValueError("must be regression or classify tree")
        self._tree_type = tree_type
        self._limit_depth = limit_depth
        self._tol_err = tol_err
        self._tol_nset = tol_nset
        self._relation = []

    def train_fit(self, X, y, relation=None):
        """ 建决策树

        :X: 训练数据
        :y: 训练数据
        :relation: 数据特征划分是标记关系, 默认回归树为"<=", 分类树为"==".
        """
        self._lables = np.lib.arraysetops.unique(y)
        if not relation:
            if self._tree_type == "reg":
                relation = ["<="] * X.shape[1]
            else:
                relation = ["=="] * X.shape[1]

        self._relation = [self.Operators[op] for op in relation]
        self._trees = self.make_tree(X, y, 1)

    def make_tree(self, X, y, tree_depth):
        if y.shape[0] < self._tol_nset or \
           self._calculate_error(y) <= self._tol_err:
            return self._create_leaf_node(y)

        (split_feature, split_value,
            less_mask, greater_mask) = self._find_best_split(X, y)
        print("split_feature =", split_feature, "split_value =", split_value)
        if split_feature is None:
            return self._create_leaf_node(y)

        t = Tree()
        t.node_val = split_value
        t.split_feature = split_feature

        if self._limit_depth and tree_depth >= self._limit_depth:
            t.is_leaf = True
            return t

        t.left = self.make_tree(X[less_mask], y[less_mask], tree_depth + 1)
        t.right = self.make_tree(
                X[greater_mask], y[greater_mask], tree_depth + 1)
        return t

    def _find_best_split(self, X, y):
        min_error, min_split_feature, min_split_value = None, None, None
        min_less_mask, min_greater_mask = None, None
        n_samples, n_features = X.shape
        for j in range(n_features):
            for split_value in set(X[:, j]):
                less_mask, greater_mask = get_split_mask(
                        X, j, split_value, self._relation[j])
                if y[less_mask].shape[0] < self._tol_nset or \
                   y[greater_mask].shape[0] < self._tol_nset:
                    continue

                error = self._calculate_error(y[less_mask]) +\
                    self._calculate_error(y[greater_mask])
                if min_error is None or error < min_error:
                    min_error = error
                    min_split_value = split_value
                    min_split_feature = j
                    min_less_mask, min_greater_mask = less_mask, greater_mask

        return min_split_feature, min_split_value,\
            min_less_mask, min_greater_mask

    def _create_leaf_node(self, y):
        t = Tree()
        t.is_leaf = True
        if self._tree_type == "reg":
            t.node_val = np.mean(y)
        else:
            c = Counter(y)
            t.node_val = c.most_common(1)[0][0]
            t.counter_samples = c
        return t

    def _calculate_gini(self, y, *args):
        n_samples = np.shape(y)[0]
        return n_samples * pGini(y, self._lables)

    def _calculate_least_squares_error(self, y, *args):
        if len(args) > 0:
            return np.sum(squared_error(y, args[0]))
        else:
            return np.sum(mean_squared_error(y))

    def _calculate_linear_model_error(self, y, *args):
        " TODO "
        pass

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i, x_row in enumerate(X):
            y[i] = self._trees.query(x_row, self._relation)
        return y

    def loss(self, test_data):
        y = self.predict(test_data[:, :-1])
        return self._calculate_error(y, test_data[:, -1])

    def prune(self, test_data):
        self.pruning(self._trees, test_data)

    def pruning(self, tree, test_data):
        if tree.is_leaf:
            return

        left_data, right_data = split_dataset(
            test_data, tree.split_feature, tree.node_val, self._relation)
        if tree.left:
            self.pruning(tree.left, left_data)
        if tree.right:
            self.pruning(tree.right, right_data)

        left = tree.left
        right = tree.right
        cal_error = self._calculate_error

        if left and left.is_leaf and right and right.is_leaf:
            errorNoMerge = cal_error(left_data[:, -1], left.node_val) + \
                cal_error(right_data[:, -1], right.node_val)
            errorMerge = cal_error(test_data[:, -1],
                                   (left.node_val + right.node_val)/2.0)

            if errorMerge < errorNoMerge:
                print("merge")
                tree.is_leaf = True
                tree.split_feature = None
                if self._tree_type == "reg":
                    tree.node_val = (left.node_val + right.node_val) / 2.0
                else:
                    counter_samples = \
                        left.counter_samples + right.counter_samples
                    tree.node_val = counter_samples.most_common(1)[0][0]
                    tree.counter_samples = counter_samples
                tree.left = None
                tree.right = None
