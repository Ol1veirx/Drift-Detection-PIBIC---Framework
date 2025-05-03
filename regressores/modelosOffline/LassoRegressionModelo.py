from sklearn.linear_model import Lasso
from regressores.ModeloBase import ModeloBase


class LassoRegressionModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.__name__ = "Lasso"
        self.modelo = Lasso(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        return self.modelo.predict(X)