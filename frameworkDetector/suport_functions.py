import numpy as np
from sklearn.metrics import mean_squared_error

from frameworkDetector.model_wrapper import ModeloWrapper
from regressores.modelosOffline.KneighborsRegressorModelo import KNeighborsRegressorModelo
from regressores.modelosOffline.RandomForestModelo import RandomForestModelo

class FunctionSuport:
    @staticmethod
    def criar_modelo(tipo="rf"):
        """Cria e retorna um ModeloWrapper com o modelo especificado."""
        if tipo == "rf":
            return ModeloWrapper(
                modelo=RandomForestModelo(n_estimators=100, random_state=42),
                nome="RandomForest",
                suporta_incremental=False
            )
        elif tipo == "knn":
            return ModeloWrapper(
                modelo=KNeighborsRegressorModelo(n_neighbors=5),
                nome="KNeighbors",
                suporta_incremental=False
            )
        else:
            raise ValueError(f"Tipo de modelo não suportado: {tipo}")

    @staticmethod
    def selecionar_melhor_modelo(pool_modelos, janela_dados, scaler):
        """Seleciona o melhor modelo do pool com base no MSE na janela."""
        if not pool_modelos or not janela_dados:
            return None

        # Prepara os dados da janela
        X_janela = np.array([x for x, _ in janela_dados])
        y_janela = np.array([y for _, y in janela_dados])
        X_janela_scaled = scaler.transform(X_janela)

        melhor_modelo = None
        menor_erro = float('inf')

        # Avalia cada modelo do pool
        for modelo in pool_modelos:
            try:
                y_pred = modelo.prever(X_janela_scaled)
                erro = mean_squared_error(y_janela, y_pred)

                if erro < menor_erro:
                    menor_erro = erro
                    melhor_modelo = modelo
            except:
                continue

        return melhor_modelo

    @staticmethod
    def adicionar_a_janela(janela, dado, tamanho_max=100):
        """Adiciona um dado à janela, removendo o mais antigo se necessário."""
        janela.append(dado)
        if len(janela) > tamanho_max:
            return janela[1:]  # Remove o mais antigo
        return janela