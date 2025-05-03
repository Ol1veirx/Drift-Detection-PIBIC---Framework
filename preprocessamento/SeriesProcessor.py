import yfinance as yf
import numpy as np
from river import preprocessing

class SeriesProcessor:
    """
    Classe para processamento de séries temporais financeiras.
    """

    @staticmethod
    def baixar_dados(symbol, periodo="5y", intervalo="1d"):
        """
        Baixa dados de séries temporais do Yahoo Finance.

        Args:
            symbol: Símbolo da ação ou índice
            periodo: Período para baixar (ex: "5y" para 5 anos)
            intervalo: Intervalo dos dados (ex: "1d" para diário)

        Returns:
            np.ndarray: Valores de fechamento
        """
        data = yf.download(symbol, period=periodo, interval=intervalo)
        return data["Close"].values

    @staticmethod
    def criar_janela_temporal(y, lags):
        """
        Cria padrões de entrada e saída para previsão de séries temporais.

        Args:
            y: Série temporal
            lags: Número de valores anteriores a usar como entrada

        Returns:
            tuple: (X, Y) onde X são os valores de entrada e Y os valores alvo
        """
        X, Y = [], []
        for i in range(len(y) - lags):
            X.append(y[i:i+lags])
            Y.append(y[i+lags])
        return np.array(X).reshape(-1, lags), np.array(Y)

    @staticmethod
    def normalizar_serie(serie_temporal: np.ndarray) -> np.ndarray:
        """
        Normaliza uma série temporal usando StandardScaler do River.

        Args:
            serie_temporal: Série temporal para normalizar

        Returns:
            np.ndarray: Série temporal normalizada
        """
        # Criando o scaler
        scaler = preprocessing.StandardScaler()

        # Aprendendo a escala com os dados
        for x in serie_temporal:
            scaler.learn_one({"valor": x[0]})

        # Transformando a série
        serie_normalizada = np.array([scaler.transform_one({"valor": x[0]})["valor"] for x in serie_temporal])

        # Garantindo que o shape permaneça correto
        return serie_normalizada.reshape(-1, 1)
