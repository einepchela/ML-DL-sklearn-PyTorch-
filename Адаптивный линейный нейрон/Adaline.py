import numpy as np

class AdalineGD:
    """Персептронный классификатор.

    Параметры
    --------------
    eta : float
        Скорость обучения (от 0.0 до 1.0)
    n_iter : int
        Количество проходов по обучающему набору
    random_state : int
        Опорное значение генератора случайных чисел для инициализации весов

    Атрибуты
    ----------
    w_ : 1d-array
        Веса после подгонки
    b_ : Scalar
        Смещение после подгонки
    errors_ : list
        Количество неправильных классификаций (обновлений) в каждой эпохе

    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Соответствие обучающим данным.

        Параметры
        ------------
        X: {array-like}, shape = [n_examples, n_features]
        Обучающий вектор, где n_examples - это количество образцов,
        а n_features - количество признаков.
        y: array-like, shape = [n_examples]
        Целевые значения

        Возвращаемые значения
        ------------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.errors_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.errors_.append(loss)
        return self

    def net_input(self, X):
        """Вычисление фактического входа"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Вычисление линейной активации"""
        return X

    def predict(self, X):
        """Возвращает метки класса после шага"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
