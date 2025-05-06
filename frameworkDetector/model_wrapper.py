import numpy as np


class ModeloWrapper:
    """Classe wrapper para modelos de regressão com interface unificada."""

    def __init__(self, modelo=None, nome=None, suporta_incremental=False):
        """Inicializa o wrapper com um modelo interno."""
        self.modelo = modelo
        self.nome = nome
        self.suporta_incremental = suporta_incremental

    def treinar(self, X, y):
        """Treina o modelo com dados fornecidos."""
        try:
            self.modelo.treinar(X, y)
            return True
        except Exception as e:
            print(f"Erro ao treinar modelo {self.nome}: {e}")
            return False

    def prever(self, X):
        """Faz previsões com o modelo."""
        return self.modelo.prever(X)

    def partial_fit(self, X, y):
        """Atualiza incrementalmente o modelo, se suportado."""
        if self.suporta_incremental:
            try:
                self.modelo.partial_fit(X, y)
                return True
            except Exception as e:
                return False
        return False