from sklearn.ensemble import RandomForestRegressor
from classes.superclasse.ModeloBase import ModeloBase

class RandomForestModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.__name__ = "Random Forest"
        self.modelo = RandomForestRegressor(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        return self.modelo.predict(X)