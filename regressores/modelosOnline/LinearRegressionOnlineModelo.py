from regressores.ModeloBase import ModeloBase
from river import linear_model

class LinearRegressionOnlineModelo(ModeloBase):
    def __init__(self):
        super().__init__()
        self.modelo = linear_model.LinearRegression()

    def treinar(self, X, y):
        for j in range(len(X)):
            X_dict = {f"t{i+1}": value for i, value in enumerate(X[j])}
            self.modelo.learn_one(X_dict, y[j][0])

    def prever(self, X):
        X_dict = {f"t{j+1}": value for j, value in enumerate(X[0])}
        return self.modelo.predict_one(X_dict)