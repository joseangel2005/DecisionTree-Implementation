"""Regression tree skeleton."""
from .base import BaseDecisionTree
import numpy as np
from .node import Node


class DecisionTreeRegressor(BaseDecisionTree):
    """Tree variant for regression.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'mse', 'mae'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """

    #criterios que acepta el arbol
    _criteria = ("mse", "mae")

    def __init__(self, max_depth=None, min_samples_split=2, criterion="mse"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
            """Recursively grow the tree.
    
            Args:
                X (ndarray, shape (n_samples, n_features)): Features at this node.
                y (ndarray, shape (n_samples,)): Targets at this node.
                depth (int): Current depth (root is 0).
    
            Returns:
                Node: Leaf (when a stopping rule triggers) or internal node with
                two subtrees built from the best split.
            """
            # create a leaf when a stopping condition holds
    
            # ``depth`` reached ``max_depth``
            if self.max_depth is not None and depth >= self.max_depth:
                return Node(value=self._leaf_value(y))
            # fewer than ``min_samples_split`` samples
            if len(y) < self.min_samples_split:
                return Node(value=self._leaf_value(y))
            # ``y`` is pure
            if len(np.unique(y)) == 1:
                return Node(value=self._leaf_value(y))
            # ``_best_split`` finds no useful split
            mejor = self._best_split(X, y)
            if mejor is None:
                return Node(value=self._leaf_value(y))
            columna, umbral = mejor
    
            # otherwise split the data on the returned (feature, threshold)
            izquierda = X[:, columna] <= umbral
            derecha = X[:, columna] > umbral
    
            # recurse on both parts with ``depth + 1``
            hijo_izq = self._build_tree(X[izquierda], y[izquierda], depth + 1)
            hijo_der = self._build_tree(X[derecha], y[derecha], depth + 1)
    
            # return the ``Node`` with its split rule and two subtrees
            return Node(feature_index=columna, threshold=umbral, left=hijo_izq, right=hijo_der)

        

    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: mean squared error or mean absolute error of y around its center (mean for 'mse', median for 'mae').

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        if self.criterion == "mse":
            # Ec. 15, var(t, D) = 1/m * sum (y_i - y_promedio)^2
            return np.mean((y - np.mean(y)) ** 2)
        return np.mean(np.abs(y - np.median(y)))

    def _best_split(self, X, y):
            """Search all features/thresholds for the largest impurity decrease.
    
            Args:
                X (ndarray, shape (n_samples, n_features)): Features at the node.
                y (ndarray, shape (n_samples,)): Targets at the node.
    
            Returns:
                tuple (int, float) or None: ``(feature_index, threshold)``, or
                None if no split reduces impurity.
            """
            # keep the split with the largest gain over ``self._impurity(y)``
            n = len(y)
            mejor_ganancia = 0.0
            mejor_split = None
            impureza_padre = self._impurity(y)
            # for each feature and candidate threshold
            for feature_index in range(X.shape[1]):
                values = np.unique(X[:, feature_index])

                # midpoints between sorted unique values
                # sacamos los umbrales y con one-hot los valores solo son 0 y 1, asi que el umbral sale en 0.5 y la regla queda como que sea esa categoria o no
                thresholds = (values[:-1] + values[1:]) / 2
                for threshold in thresholds:
                    # partition ``y`` into left (``x <= threshold``) and right
                    filtro = X[:, feature_index] <= threshold
                    y_izq = y[filtro]
                    y_der = y[~filtro]
                    # skip splits leaving a side empty
                    if len(y_izq) == 0 or len(y_der) == 0:
                        continue
                    # compute the weighted impurity ``(n_l * imp_l + n_r * imp_r) / n``
                    # Ec. 11, rem(d, D) = sum |D_d=l| / |D| * impureza(D_d=l)
                    resto = (len(y_izq) * self._impurity(y_izq) + len(y_der) * self._impurity(y_der)) / n
                    # Ec. 12, IG(d, D) = impureza(D) - rem(d, D)
                    ganancia = impureza_padre - resto
                    if ganancia > mejor_ganancia:
                        mejor_ganancia = ganancia
                        mejor_split = (feature_index, float(threshold))
            return mejor_split

    def _leaf_value(self, y):
        """Return the mean of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """
        return np.mean(y)

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: R^2 = 1 - SS_res / SS_tot.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

