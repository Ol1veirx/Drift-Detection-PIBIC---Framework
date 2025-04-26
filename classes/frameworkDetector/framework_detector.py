import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import traceback
import copy

# Importar wrappers de modelo
from classes.modelosOffline.KneighborsRegressorModelo import KNeighborsRegressorModelo
from classes.modelosOffline.LassoRegressionModelo import LassoRegressionModelo
from classes.modelosOffline.LinearRegressionModelo import LinearRegressionModelo
from classes.modelosOffline.RandomForestModelo import RandomForestModelo
from classes.modelosOffline.RidgeRegressionModelo import RidgeRegressionModelo
from classes.modelosOffline.SVRModelo import SVRModelo

class FrameworkDetector:
    """
    Classe utilitária estática para o framework de detecção e adaptação.
    """

    @staticmethod
    def treinar_modelo_inicial(X, y, tipo_modelo=RandomForestModelo):
        """Treina o modelo inicial e ajusta o scaler."""
        modelo_treinado = None
        scaler = StandardScaler()

        # Ajusta e aplica o scaler
        print("Ajustando Scaler nos dados iniciais...")
        X_scaled = scaler.fit_transform(X)
        print("✓ Scaler ajustado e aplicado.")

        # Instancia e treina o modelo inicial
        nome_modelo = tipo_modelo.__name__
        print(f"\nIniciando treinamento do modelo inicial: {nome_modelo}...")
        modelo_wrapper_instancia = tipo_modelo()
        modelo_wrapper_instancia.treinar(X_scaled, y)

        if modelo_wrapper_instancia.modelo is not None:
            print(f"  ✅ Modelo {nome_modelo} treinado com sucesso")
            modelo_treinado = modelo_wrapper_instancia
        else:
            print(f"  ❌ Falha ao treinar modelo {nome_modelo}")

        # Fallback simples
        if modelo_treinado is None:
            print("\n⚠️ AVISO: Modelo inicial não treinado. Tentando Regressão Linear como fallback...")
            modelo_fallback_wrapper = LinearRegressionModelo()
            modelo_fallback_wrapper.treinar(X_scaled, y)
            if modelo_fallback_wrapper.modelo is not None:
                modelo_treinado = modelo_fallback_wrapper
                print("  ✅ Modelo de fallback (Regressão Linear) treinado com sucesso.")
            else:
                 print("  ❌ Erro crítico: Não foi possível treinar o modelo de fallback.")

        return modelo_treinado, scaler

    @staticmethod
    def treinar_novo_conceito(X_novo, y_novo, scaler, tipo_modelo=RandomForestModelo):
        """Treina um novo modelo com os dados do novo conceito."""
        modelo_novo_conceito = None
        nome_modelo = tipo_modelo.__name__
        print(f"  Treinando novo modelo ({nome_modelo}) com {len(X_novo)} amostras do novo conceito...")

        if len(X_novo) == 0:
            print("  ❌ Erro: Nenhum dado fornecido para treinar o novo conceito.")
            return None

        # Escala e treina o novo modelo
        X_novo_scaled = scaler.transform(X_novo)
        modelo_wrapper_instancia = tipo_modelo()
        modelo_wrapper_instancia.treinar(X_novo_scaled, y_novo)

        if modelo_wrapper_instancia.modelo is not None:
            print(f"  ✅ Novo modelo ({nome_modelo}) treinado com sucesso.")
            modelo_novo_conceito = modelo_wrapper_instancia
        else:
            print(f"  ❌ Falha ao treinar modelo {nome_modelo} para novo conceito.")

        return modelo_novo_conceito


    @staticmethod
    def selecionar_melhor_modelo(pool_modelos, janela_dados, scaler):
        """Seleciona o melhor modelo do pool com base no MSE na janela."""
        if not pool_modelos:
            print("⚠️ AVISO: Pool de modelos vazio para seleção!")
            return None
        if not janela_dados:
            print("⚠️ AVISO: Janela de dados vazia para seleção!")
            return None

        # Prepara os dados da janela
        X_janela_list = [x for x, y in janela_dados]
        y_janela = np.array([y for x, y in janela_dados])
        X_janela_scaled = scaler.transform(np.array(X_janela_list))

        melhor_modelo = None
        menor_erro = float('inf')

        # Avalia cada modelo do pool na janela
        print(f"  Avaliando {len(pool_modelos)} modelos do pool na janela ({len(y_janela)} amostras)...")
        for modelo_wrapper in pool_modelos:
            y_pred = modelo_wrapper.prever(X_janela_scaled)
            erro = mean_squared_error(y_janela, y_pred)

            if erro < menor_erro:
                menor_erro = erro
                melhor_modelo = modelo_wrapper

        # Retorna o melhor modelo encontrado
        if melhor_modelo is None:
            print("  ERRO INESPERADO: Nenhum modelo pôde ser avaliado ou selecionado!")
            return None
        else:
            print(f"  ✓ Melhor modelo do pool selecionado: {melhor_modelo.nome if hasattr(melhor_modelo, 'nome') else type(melhor_modelo).__name__} (MSE: {menor_erro:.4f})")
            return melhor_modelo

    @staticmethod
    def desempenho(modelo_wrapper, janela_dados, scaler):
        """Avalia o desempenho (MSE) de um modelo na janela."""
        if not janela_dados: return float('inf')

        # Prepara, escala e prevê com os dados da janela
        X_janela_list = [x for x, y in janela_dados]
        y_janela = np.array([y for x, y in janela_dados])
        X_janela_scaled = scaler.transform(np.array(X_janela_list))
        y_pred = modelo_wrapper.prever(X_janela_scaled)

        return mean_squared_error(y_janela, y_pred)

    @staticmethod
    def adicionar_a_janela(janela, dado, tamanho_max=100):
        """Adiciona um dado à janela, mantendo o tamanho máximo."""
        janela.append(dado)
        if len(janela) > tamanho_max:
            return janela[1:]
        return janela

    @staticmethod
    def get_state(detector_wrapper):
        """Obtém o estado ('NORMAL', 'ALERTA', 'MUDANÇA') do detector."""
        # Tenta verificar o estado de drift primeiro
        if hasattr(detector_wrapper, 'drift_detectado'):
            return "MUDANÇA"

        # Tenta verificar o estado de alerta (para detectores River)
        river_detector = getattr(detector_wrapper, 'detector', None)
        if river_detector:
             return "ALERTA"

        # Se nenhum dos anteriores, assume NORMAL
        return "NORMAL"
