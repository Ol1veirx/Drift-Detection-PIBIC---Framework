from river import linear_model
from regressores.ModeloBase import ModeloBase


class BayesianLinearRegressionModelo(ModeloBase):
    def __init__(self):
        super().__init__()
        self.modelo = linear_model.BayesianLinearRegression()

    def treinar(self, X, y):
        for i in range(len(X)):
            X_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}
            self.modelo.learn_one(X_dict, y[i][0])

    def prever(self, X):
        X_dict = {f"t{j+1}": value for j, value in enumerate(X[0])}  # Converte X[0] em dicionário
        return self.modelo.predict_one(X_dict)
