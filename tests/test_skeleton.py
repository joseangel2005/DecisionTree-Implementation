import pytest

from decision_tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


def test_base_is_abstract():
    """Revisa que no se pueda crear un BaseDecisionTree solo.

    Es una clase abstracta, o sea una plantilla, porque tiene metodos marcados
    con @abstractmethod que ella no implementa. Python no deja crear objetos de
    una clase asi, y por eso aqui esperamos un TypeError. Las que si se pueden
    usar son DecisionTreeClassifier y DecisionTreeRegressor, que son las que
    heredan de ella y si implementan esos metodos.
    """
    with pytest.raises(TypeError):
        BaseDecisionTree()


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_hyperparameters(cls):
    """Revisa que get_params y set_params funcionen en los dos arboles.

    get_params nos tiene que regresar los hiperparametros que le pasamos al
    crear el modelo, set_params nos tiene que dejar cambiarlos despues, y si le
    mandamos un nombre que no existe como foo nos tiene que marcar ValueError.
    Lo corremos con los dos arboles porque los dos heredan estos metodos de
    BaseDecisionTree.
    """
    model = cls(max_depth=3, min_samples_split=4)
    assert model.get_params()["max_depth"] == 3
    model.set_params(criterion="entropy")
    assert model.criterion == "entropy"
    with pytest.raises(ValueError):
        model.set_params(foo=1)

# para los tests de abajo, usamos un dataset de 6 filas y 4 columnas, donde la
X = [
    [1.0, 1, 0, 0],
    [2.0, 1, 0, 0],
    [3.0, 0, 1, 0],
    [4.0, 0, 1, 0],
    [5.0, 0, 0, 1],
    [6.0, 0, 0, 1],
]



@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classifier_categorical(criterion):
    """Revisa que el clasificador sepa partir con atributos categoricos.

    Las etiquetas son b justo en las dos filas que traen 1 en la columna 2, que
    es la categoria verde. Entonces la mejor particion posible es preguntar si
    es verde o no, y el arbol tiene que escoger esa columna, por eso pedimos que
    root.feature_index sea 2. Y como separa perfecto, el accuracy debe dar 1.0.
    """
    y = ["a", "a", "b", "b", "a", "a"]
    model = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    assert model.eval(X, y) == 1.0
    assert model.root.feature_index == 2


def test_classifier_numeric():
    """Revisa que el clasificador sepa partir con atributos numericos.

    Las etiquetas son 0 en las tres primeras filas y 1 en las tres ultimas, asi
    que el corte perfecto va entre el 3 y el 4 de la columna 0. El umbral debe
    salir en 3.5, que es el promedio de ese par de valores

    """
    y = [0, 0, 0, 1, 1, 1]
    model = DecisionTreeClassifier().fit(X, y)
    assert model.root.feature_index == 0
    assert model.root.threshold == 3.5
    assert list(model.predict([[0.5, 0, 0, 0], [10.0, 0, 0, 0]])) == [0, 1]


@pytest.mark.parametrize("criterion", ["mse", "mae"])
def test_regressor(criterion):
    """Revisa que el arbol de regresion prediga valores numericos.

    Aqui las etiquetas ya no son clases sino numeros, y cada categoria del
    one-hot tiene su propio valor, 10 para rojo, 20 para verde y 30 para azul.
    El arbol tiene que separar las tres y dejar en cada hoja el promedio que le
    toca, asi que la R2 debe dar 1.0. Lo corremos con mse y con mae para probar
    los dos criterios.
    """
    y = [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
    model = DecisionTreeRegressor(criterion=criterion).fit(X, y)
    assert model.eval(X, y) == 1.0


def test_max_depth_zero_is_leaf():
    """Revisa el caso limite de max_depth en cero.

    Con max_depth igual a 0 el arbol no tiene permitido partir ni una vez, asi
    que debe quedar nada mas la raiz convertida en hoja, aunque los datos si se
    pudieran separar. Por eso se pidio que root.is_leaf sea True.
    """
    y = [0, 0, 0, 1, 1, 1]
    model = DecisionTreeClassifier(max_depth=0).fit(X, y)
    assert model.root.is_leaf


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_invalid_params(cls):
    """Revisa que fit valide los hiperparametros y que predict pida entrenar.

    Son los tres errores que pusimos en base.py. Un criterio que no existe como
    foo, un min_samples_split de 1 cuando el minimo es 2, y llamar predict antes
    de fit, cuando todavia no hay arbol. Los tres deben marcar ValueError, y lo
    probamos en los dos arboles.
    """
    with pytest.raises(ValueError):
        cls(criterion="foo").fit(X, [0] * 6)
    with pytest.raises(ValueError):
        cls(min_samples_split=1).fit(X, [0] * 6)
    with pytest.raises(ValueError):
        cls().predict(X)