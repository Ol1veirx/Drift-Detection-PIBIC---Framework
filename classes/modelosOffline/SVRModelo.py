from sklearn.svm import SVR
from classes.superclasse.ModeloBase import ModeloBase


class SVRModelo(ModeloBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.modelo = SVR(**kwargs)

    def treinar(self, X, y):
        self.modelo.fit(X, y)
        return self

    def prever(self, X):
        return self.modelo.predict(X)