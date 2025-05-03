import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import traceback
import copy
from sklearn.cluster import KMeans

# Importar wrappers de modelo
from regressores.modelosOffline.KneighborsRegressorModelo import KNeighborsRegressorModelo
from regressores.modelosOffline.LassoRegressionModelo import LassoRegressionModelo
from regressores.modelosOffline.LinearRegressionModelo import LinearRegressionModelo
from regressores.modelosOffline.RandomForestModelo import RandomForestModelo
from regressores.modelosOffline.RidgeRegressionModelo import RidgeRegressionModelo
from regressores.modelosOffline.SVRModelo import SVRModelo

class FrameworkDetector:
    """
    Classe utilitária estática para o framework de detecção e adaptação.
    """
    _pools_por_regime = {}
    _modelos_por_regime = {}

    @staticmethod
    def treinar_modelo_inicial(X, y, tipo_modelo=RandomForestModelo):
        """Treina o modelo inicial e ajusta o scaler."""
        modelo_treinado = None
        scaler = StandardScaler()

        print("Ajustando Scaler nos dados iniciais...")
        X_scaled = scaler.fit_transform(X)
        print("✓ Scaler ajustado e aplicado.")

        nome_modelo = tipo_modelo.__name__
        print(f"\nIniciando treinamento do modelo inicial: {nome_modelo}...")
        modelo_wrapper_instancia = tipo_modelo()
        modelo_wrapper_instancia.treinar(X_scaled, y)

        if modelo_wrapper_instancia.modelo is not None:
            print(f"  Modelo {nome_modelo} treinado com sucesso")
            modelo_treinado = modelo_wrapper_instancia
        else:
            print(f"  ❌ Falha ao treinar modelo {nome_modelo}")

        if modelo_treinado is None:
            print("\nAVISO: Modelo inicial não treinado. Tentando Regressão Linear como fallback...")
            modelo_fallback_wrapper = LinearRegressionModelo()
            modelo_fallback_wrapper.treinar(X_scaled, y)
            if modelo_fallback_wrapper.modelo is not None:
                modelo_treinado = modelo_fallback_wrapper
                print("  Modelo de fallback (Regressão Linear) treinado com sucesso.")
            else:
                 print("  Erro crítico: Não foi possível treinar o modelo de fallback.")

        return modelo_treinado, scaler

    @staticmethod
    def treinar_novo_conceito(X_novo, y_novo, scaler, tipo_modelo=RandomForestModelo):
        """Treina um novo modelo com os dados do novo conceito."""
        modelo_novo_conceito = None
        nome_modelo = tipo_modelo.__name__
        print(f"  Treinando novo modelo ({nome_modelo}) com {len(X_novo)} amostras do novo conceito...")

        if len(X_novo) == 0:
            print("  Erro: Nenhum dado fornecido para treinar o novo conceito.")
            return None

        # Escala e treina o novo modelo
        X_novo_scaled = scaler.transform(X_novo)
        modelo_wrapper_instancia = tipo_modelo()
        modelo_wrapper_instancia.treinar(X_novo_scaled, y_novo.ravel())

        if modelo_wrapper_instancia.modelo is not None:
            print(f"  Novo modelo ({nome_modelo}) treinado com sucesso.")
            modelo_novo_conceito = modelo_wrapper_instancia
        else:
            print(f"  Falha ao treinar modelo {nome_modelo} para novo conceito.")

        return modelo_novo_conceito


    @staticmethod
    def selecionar_melhor_modelo(pool_modelos, janela_dados, scaler):
        """Seleciona o melhor modelo do pool com base no MSE na janela."""
        if not pool_modelos:
            print("AVISO: Pool de modelos vazio para seleção!")
            return None
        if not janela_dados:
            print("AVISO: Janela de dados vazia para seleção!")
            return None

        X_janela_list = [x for x, y in janela_dados]
        y_janela = np.array([y for x, y in janela_dados])
        X_janela_scaled = scaler.transform(np.array(X_janela_list))

        melhor_modelo = None
        menor_erro = float('inf')

        print(f"  Avaliando {len(pool_modelos)} modelos do pool na janela ({len(y_janela)} amostras)...")
        for modelo_wrapper in pool_modelos:
            y_pred = modelo_wrapper.prever(X_janela_scaled)
            erro = mean_squared_error(y_janela, y_pred)

            if erro < menor_erro:
                menor_erro = erro
                melhor_modelo = modelo_wrapper

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
        if hasattr(detector_wrapper, 'drift_detectado') and detector_wrapper.drift_detectado:
               return "MUDANÇA"

        if hasattr(detector_wrapper, 'is_warning_detected') and detector_wrapper.is_warning_detected:
            return "ALERTA"

        river_detector = getattr(detector_wrapper, 'detector', None)
        if river_detector and hasattr(river_detector, '_warning_detected') and river_detector._warning_detected:
            return "ALERTA"

        return "NORMAL"

    @staticmethod
    def adicionar_ao_pool(pool_modelos, novo_modelo, max_pool_size=5,
                        janela_dados=None, scaler=None,
                        min_diversidade_erro=0.05):
        """
        Adiciona um modelo ao pool considerando diversidade e qualidade.

        Args:
            pool_modelos: Lista atual de modelos
            novo_modelo: Modelo a ser potencialmente adicionado
            max_pool_size: Tamanho máximo do pool
            janela_dados: Dados recentes para avaliação
            scaler: Scaler para preparar os dados
            min_diversidade_erro: Diferença mínima de MSE para considerar modelo diverso

        Returns:
            pool_modelos atualizado
        """
        if not pool_modelos:
            pool_modelos.append(novo_modelo)
            return pool_modelos

        if janela_dados and len(janela_dados) > 5 and scaler:
            erro_novo = FrameworkDetector.desempenho(novo_modelo, janela_dados, scaler)

            modelo_similar = None
            for modelo in pool_modelos:
                erro_modelo = FrameworkDetector.desempenho(modelo, janela_dados, scaler)
                if abs(erro_modelo - erro_novo) < min_diversidade_erro:
                    modelo_similar = modelo
                    break

            if modelo_similar:
                if erro_novo < FrameworkDetector.desempenho(modelo_similar, janela_dados, scaler):
                    print(f"  Substituindo modelo similar de qualidade inferior no pool")
                    pool_modelos.remove(modelo_similar)
                    pool_modelos.append(novo_modelo)
                else:
                    print(f"  ℹNão adicionando modelo: similar a um existente e não melhor")
                    return pool_modelos
            else:
                pool_modelos.append(novo_modelo)
        else:
            pool_modelos.append(novo_modelo)

        if len(pool_modelos) > max_pool_size:
            if janela_dados and len(janela_dados) > 5 and scaler:
                pior_modelo = None
                pior_erro = float('-inf')

                for modelo in pool_modelos:
                    erro = FrameworkDetector.desempenho(modelo, janela_dados, scaler)
                    if erro > pior_erro:
                        pior_erro = erro
                        pior_modelo = modelo

                if pior_modelo:
                    pool_modelos.remove(pior_modelo)
                    print(f"  🗑️ Removido o pior modelo do pool (MSE: {pior_erro:.4f})")
                    return pool_modelos

            modelo_removido = pool_modelos.pop(0)
            print(f"  Removido o modelo mais antigo do pool")

        return pool_modelos

    @staticmethod
    def identificar_regime(janela_dados, n_clusters=3):
        """Identifica o regime atual baseado na janela de dados recentes."""
        if not janela_dados or len(janela_dados) < n_clusters:
            return 0

        X = np.array([x for x, _ in janela_dados])
        y = np.array([y for _, y in janela_dados])

        features_mean = np.mean(X, axis=0)
        features_var = np.var(X, axis=0)
        target_mean = np.mean(y)
        target_var = np.var(y)

        regime_features = np.concatenate([features_mean, features_var, [target_mean, target_var]])

        if not hasattr(FrameworkDetector, '_regime_samples'):
            FrameworkDetector._regime_samples = []

        # Adicionar amostra
        FrameworkDetector._regime_samples.append(regime_features)

        if len(FrameworkDetector._regime_samples) < n_clusters:
            return 0

        samples = np.array(FrameworkDetector._regime_samples)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(samples)

        # Prever o regime atual
        regime = kmeans.predict([regime_features])[0]
        return int(regime)

    @staticmethod
    def obter_melhor_modelo_para_regime(janela_dados, scaler, pools_por_regime, regime_atual):
        """Obtém o melhor modelo para o regime atual."""
        if regime_atual not in pools_por_regime or not pools_por_regime[regime_atual]:
            if 0 in pools_por_regime and pools_por_regime[0]:
                regime_busca = 0
            else:
                return None
        else:
            regime_busca = regime_atual

        pool_do_regime = pools_por_regime[regime_busca]
        melhor_modelo = FrameworkDetector.selecionar_melhor_modelo(
            pool_do_regime, janela_dados, scaler)

        print(f"  Avaliando pool do regime {regime_busca} ({len(pool_do_regime)} modelos)")
        return melhor_modelo

    @staticmethod
    def gerenciar_pool(pool_modelos, janela_dados, scaler, estado_detector,
                      modelo_atual, max_pool_size=5):
        """
        Gerencia inteligentemente o pool baseado no estado do detector.
        """
        if estado_detector == "ALERTA":
            if hasattr(modelo_atual, "criar_variante"):
                variante = modelo_atual.criar_variante(mutation_strength=0.2)
                if variante:
                    FrameworkDetector.adicionar_ao_pool(
                        pool_modelos, variante, max_pool_size,
                        janela_dados, scaler, min_diversidade_erro=0.03)
                    print(f"  Variante do modelo atual adicionada ao pool em estado de ALERTA")

        elif estado_detector == "MUDANÇA":
            melhor = FrameworkDetector.selecionar_melhor_modelo(pool_modelos, janela_dados, scaler)
            if melhor:
                novos_modelos = [melhor]
                for modelo in pool_modelos[-max_pool_size//2:]:
                    if modelo is not melhor:
                        erro = FrameworkDetector.desempenho(modelo, janela_dados, scaler)
                        erro_melhor = FrameworkDetector.desempenho(melhor, janela_dados, scaler)
                        if erro < erro_melhor * 1.5:
                            novos_modelos.append(modelo)
                print(f"  Pool reconfigurado após MUDANÇA: {len(novos_modelos)}/{len(pool_modelos)} modelos mantidos")
                return novos_modelos

        return pool_modelos