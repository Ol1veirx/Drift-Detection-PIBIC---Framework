
from river import neighbors
from regressores.ModeloBase import ModeloBase

class KNNRegressorOnlineModelo(ModeloBase):
    def __init__(self, n_neighbors=10, aggregation_method="mean"):
        super().__init__()
        self.modelo = neighbors.KNNRegressor(n_neighbors=n_neighbors, aggregation_method=aggregation_method)

    def treinar(self, X, y):
        for i in range(len(X)):
            X_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}
            self.modelo.learn_one(X_dict, y[i][0])

    def prever(self, X):
        X_dict = {f"t{i+1}": value for i, value in enumerate(X[0])}
        return self.modelo.predict_one(X_dict)  # Retorna o valor escalar diretamente