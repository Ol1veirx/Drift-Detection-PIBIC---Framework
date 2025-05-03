from river import tree
from regressores.ModeloBase import ModeloBase

class HoeffdingTreeRegressorModelo(ModeloBase):
    def __init__(self, grace_period=100, leaf_prediction="adaptive"):
        super().__init__()
        self.modelo = tree.HoeffdingTreeRegressor()

    def treinar(self, X, y):
        for i in range(len(X)):
            X_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}
            self.modelo.learn_one(X_dict, y[i][0])

    def prever(self, X):
        X_dict = {f"t{i+1}": value for i, value in enumerate(X[0])}
        return self.modelo.predict_one(X_dict)