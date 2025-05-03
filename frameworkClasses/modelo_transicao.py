class ModeloTransicao:
    def __init__(self, modelo):
        self.modelo = modelo
        self.nome = "ModeloTransicao"

    def prever(self, X):
        return self.modelo.prever(X)