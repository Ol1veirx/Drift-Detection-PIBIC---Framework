import numpy as np


class ModelWrapper:
    """Wrapper para modelos de ML para usar no framework de detecção de drift"""

    def __init__(self, modelo, nome="Modelo"):
        self.modelo = modelo
        self.nome = nome

    def prever(self, X):
        """Realiza predição no formato numpy array"""
        # Garantir que X está no formato correto
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.modelo.predict(X)

    def treinar(self, X, y):
        """Treina o modelo com os dados fornecidos"""
        X = np.array(X)
        y = np.array(y)
        self.modelo.fit(X, y)
        return self

    def partial_fit(self, X, y):
        """Treino incremental, se suportado pelo modelo"""
        if hasattr(self.modelo, 'partial_fit'):
            X = np.array(X)
            y = np.array(y)
            self.modelo.partial_fit(X, y)
        return self