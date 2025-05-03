import numpy as np
from regressores.ModeloBase import ModeloBase

class ModelTrainer:
    """
    Classe para treinamento de modelos.
    """

    @staticmethod
    def inicializar_modelos(modelo_classe, detector_classe, **kwargs):
        """
        Inicializa instâncias de modelo e detector.

        Args:
            modelo_classe: Classe do modelo
            detector_classe: Classe do detector de drift
            **kwargs: Parâmetros para inicialização

        Returns:
            tuple: (modelo_instancia, detector_instancia)
        """
        # Instancia o modelo com os parâmetros fornecidos
        modelo_instancia = modelo_classe(**kwargs)

        # Instancia o detector com os parâmetros fornecidos
        detector_instancia = detector_classe(**kwargs)

        return modelo_instancia, detector_instancia

    @staticmethod
    def treinamento_modelo_batch(modelo, X, y):
        """
        Treina um modelo em modo batch e calcula erro médio.

        Args:
            modelo: Instância do modelo a ser treinado
            X: Dados de entrada
            y: Valores alvo

        Returns:
            float: Erro médio de treinamento
        """
        # Treinamento do modelo usando o método 'treinar' da subclasse
        modelo.treinar(X, y)

        # Cálculo do erro médio (adapte para modelos online, se necessário)
        if isinstance(modelo, ModeloBase):  # Verifica se é uma instância da superclasse dos modelos offline
            erro_medio = np.abs(np.mean(y - modelo.prever(X))) # Calcula erro para modelos offline
        else: # Senão calcula para modelos online
            predicoes = []
            for i in range(len(X)):
                predicoes.append(modelo.prever(X[i].reshape(1, -1))[0]) # Faz as predições para cada exemplo em X
            erro_medio = np.abs(np.mean(y.ravel() - np.array(predicoes))) # Calcula o erro médio

        return erro_medio

    @staticmethod
    def treinamento_online_many(modelo, X, y, tamanho_batch):
        """
        Treina um modelo online com um conjunto inicial de dados.

        Args:
            modelo: Modelo online a ser treinado
            X: Dados de entrada
            y: Valores alvo
            tamanho_batch: Tamanho do lote inicial para treinamento

        Returns:
            modelo: Modelo treinado
        """
        # Treina o modelo com os primeiros 'tamanho_batch' exemplos
        modelo.treinar(X[:tamanho_batch], y[:tamanho_batch])
        return modelo
