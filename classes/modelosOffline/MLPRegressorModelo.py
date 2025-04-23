from sklearn.neural_network import MLPRegressor
from classes.superclasse.ModeloBase import ModeloBase

class MLPRegressorModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.modelo = MLPRegressor(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        return self.modelo.predict(X)