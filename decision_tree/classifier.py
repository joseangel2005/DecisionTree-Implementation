"""Classification tree skeleton."""
from .base import BaseDecisionTree
import numpy as np
from .node import Node

class DecisionTreeClassifier(BaseDecisionTree):
    """Tree variant for classification.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'gini', 'entropy'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """
    #criterios que acepta el arbol
    _criteria = ("gini", "entropy")

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
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
        """Impurity of ``y`` per ``self.criterion``: Gini impurity 1 - sum(p_k^2) or entropy -sum(p_k * log2(p_k)), where p_k is the fraction of samples of class k.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        # implement the measure selected by ``self.criterion``
        # where p_k is the fraction of samples of class k
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        if self.criterion == "gini":
            # Ec. 14, Gini(t, D) = 1 - sum P(t = l)^2
            return 1 - np.sum(p ** 2)
        # Ec. 9, H(t) = - sum P(t = i) * log2(P(t = i))
        return -np.sum(p * np.log2(p))

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
        """Return the majority (most frequent) class of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """

        # contamos cuantos hay de cada clase y nos quedamos con la que mas se repite
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: accuracy = fraction of correct predictions.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
       # comparamos las dos listas y nos da puros True y False, el promedio de eso es la fraccion de aciertos
        return np.mean(y_true == y_pred)
