import numpy as np
from sklearn.metrics import mean_squared_error

from classes.modelosOffline.KneighborsRegressorModelo import KNeighborsRegressorModelo
from classes.modelosOffline.LassoRegressionModelo import LassoRegressionModelo
from classes.modelosOffline.LinearRegressionModelo import LinearRegressionModelo
from classes.modelosOffline.RandomForestModelo import RandomForestModelo
from classes.modelosOffline.RidgeRegressionModelo import RidgeRegressionModelo
from classes.modelosOffline.SVRModelo import SVRModelo
from utils.SeriesProcessor import SeriesProcessor


class FrameworkDetector:
    @staticmethod
    def treinar_modelos_iniciais(X, y):
        """Treina e retorna uma lista de modelos iniciais"""
        modelos = [
            LinearRegressionModelo().treinar(X, y),
            KNeighborsRegressorModelo().treinar(X, y),
            LassoRegressionModelo().treinar(X, y),
            #MLPRegressorModelo.treinar(X, y),  # Comentado por ser mais lento
            RandomForestModelo().treinar(X, y),
            RidgeRegressionModelo().treinar(X, y),
            SVRModelo().treinar(X, y)
        ]
        return modelos

    @staticmethod
    def selecionar_melhor_modelo(pool, janela):
        """Seleciona o melhor modelo do pool com base no erro quadrático médio"""
        X_janela = [x for x, y in janela]
        y_janela = [y for x, y in janela]

        erros = [mean_squared_error(y_janela, modelo.prever(X_janela)) for modelo in pool]
        return pool[np.argmin(erros)]

    @staticmethod
    def desempenho(modelo, janela):
        """Avalia o desempenho do modelo na janela atual"""
        X_janela = [x for x, y in janela]
        y_janela = [y for x, y in janela]
        return mean_squared_error(y_janela, modelo.prever(X_janela))

    @staticmethod
    def adicionar_a_janela(janela, dado, tamanho_max=100):
        """Adiciona um novo dado à janela, removendo o mais antigo se necessário"""
        janela.append(dado)
        if len(janela) > tamanho_max:
            janela.pop(0)
        return janela  # É uma boa prática retornar a janela atualizada

    def get_state(detector):
        """
        Função auxiliar para mapear o estado do detector para NORMAL, ALERTA ou MUDANÇA.
        Funciona com diferentes tipos de detectores.
        """
        # DDM
        if hasattr(detector, 'in_concept_change'):
            if detector.in_concept_change:
                return "MUDANÇA"
            elif detector.in_warning_zone:
                return "ALERTA"
            else:
                return "NORMAL"

        # ADWIN
        elif hasattr(detector, 'detected_change'):
            if detector.detected_change():
                return "MUDANÇA"
            # ADWIN geralmente não tem estado de alerta, então verificamos a magnitude da diferença
            elif hasattr(detector, 'estimation'):
                # Esta é uma implementação aproximada - você precisará ajustar baseado na sua implementação
                if abs(detector.estimation - detector._last_estimation) > detector.delta / 2:
                    return "ALERTA"
            return "NORMAL"

        # ECDD ou outros detectores
        elif hasattr(detector, '_current_state') or hasattr(detector, 'current_state'):
            state = getattr(detector, '_current_state', None) or getattr(detector, 'current_state', None)
            if state in ["NORMAL", "ALERTA", "MUDANÇA"]:
                return state
            # Mapear outros estados se necessário

        # Caso genérico - tenta várias abordagens comuns
        else:
            # Tentar acessar variáveis/métodos comuns que indicam mudança
            # Estas são apenas sugestões - ajuste conforme suas implementações específicas

            # Verificar se há métodos específicos
            if hasattr(detector, 'detected_drift') and callable(getattr(detector, 'detected_drift')):
                if detector.detected_drift():
                    return "MUDANÇA"

            if hasattr(detector, 'warning_detected') and callable(getattr(detector, 'warning_detected')):
                if detector.warning_detected():
                    return "ALERTA"

            # Verificar atributos comuns
            if hasattr(detector, 'drift_detected') and detector.drift_detected:
                return "MUDANÇA"

            if hasattr(detector, 'warning') and detector.warning:
                return "ALERTA"

        # Se nenhuma das verificações anteriores funcionar, assume estado normal
        return "NORMAL"

    def calcular_erro_relativo(y_true, y_pred):
        """Calcula o erro relativo médio absoluto"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))

    # 1. Dados Sintéticos com Drift Controlado
    def gerar_dados_sinteticos(n_amostras=5000, n_features=5, n_drifts=3, seed=None):
        """Gera dados sintéticos com drift controlado para regressão"""
        if seed is not None:
            np.random.seed(seed)

        X = np.random.randn(n_amostras, n_features)
        y = np.zeros(n_amostras)

        # Pontos de drift
        drift_points = [int(i * n_amostras / (n_drifts + 1)) for i in range(1, n_drifts + 1)]

        # Regras para cada segmento
        def regra_1(x):
            return 5 + 2 * x[:, 0] + 0.5 * x[:, 1] + np.random.randn(len(x)) * 0.5

        def regra_2(x):
            return 10 - 1.5 * x[:, 0] + 1 * x[:, 1] + np.random.randn(len(x)) * 0.8

        def regra_3(x):
            return 2 + 3 * x[:, 0] + 2 * x[:, 2] + np.random.randn(len(x)) * 0.3

        def regra_4(x):
            return 8 + 0.1 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * x[:, 3] + np.random.randn(len(x)) * 0.6

        regras = [regra_1, regra_2, regra_3, regra_4]

        # Aplicar regras aos segmentos
        regra_atual = 0
        inicio = 0

        for ponto in drift_points:
            # Aplicar regra atual até o ponto de drift
            indices = np.arange(inicio, ponto)
            y[indices] = regras[regra_atual](X[indices])

            # Mudar para próxima regra
            regra_atual = (regra_atual + 1) % len(regras)
            inicio = ponto

        # Aplicar regra final no último segmento
        indices = np.arange(inicio, n_amostras)
        y[indices] = regras[regra_atual](X[indices])

        return X, y, drift_points
