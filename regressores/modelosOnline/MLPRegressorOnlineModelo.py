from regressores.ModeloBase import ModeloBase
from river import neural_net
from river import optim


class MLPRegressorOnlineModelo(ModeloBase):
    def __init__(self, hidden_dims=(20,), activations=("ReLU",), optimizer=optim.Adam(0.001)):
        super().__init__()
        self.modelo = neural_net.MLPRegressor(hidden_dims=hidden_dims, activations=activations, optimizer=optimizer)

    def treinar(self, X, y):
        for i in range(len(X)):
            X_dict = {f"t{j+1}": value for j, value in enumerate(X[i])}
            self.modelo.learn_one(X_dict, y[i][0])

    def prever(self, X):
        X_dict = {f"t{i+1}": value for i, value in enumerate(X[0])}
        return self.modelo.predict_one(X_dict)  # Retorna o valor escalar diretamente