from sklearn.neighbors import KNeighborsRegressor
from classes.superclasse.ModeloBase import ModeloBase


class KNeighborsRegressorModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.__name__ = "KNN"
        self.modelo = KNeighborsRegressor(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        return self.modelo.predict(X)