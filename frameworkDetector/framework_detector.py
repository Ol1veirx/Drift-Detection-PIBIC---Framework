import time
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.cluster import KMeans
from tqdm import tqdm

from frameworkDetector.detector_wrapper import DetectorWrapper
from frameworkDetector.suport_functions import FunctionSuport

class FrameworkDetector:
    """
    Classe utilitária estática para o framework de detecção e adaptação,
    com uma noção simplificada de regimes.
    """

    @staticmethod
    def _inicializar_componentes(X_init, y_init, tipo_modelo, detector_externo, tamanho_janela):
        """Inicializa o scaler, modelo inicial, pools, detector e janelas."""
        scaler = StandardScaler()
        if X_init.shape[0] > 0:
            X_init_scaled = scaler.fit_transform(X_init)
        else:
            X_init_scaled = X_init

        modelo_inicial = FunctionSuport.criar_modelo(tipo_modelo)
        if X_init.shape[0] > 0:
            modelo_inicial.treinar(X_init_scaled, y_init)
        print(f"Modelo inicial ({modelo_inicial.nome}) treinado com {len(X_init)} amostras.")

        regime_atual_id = 0
        modelo_atual = modelo_inicial
        pools_por_regime = {regime_atual_id: [modelo_inicial] if X_init.shape[0] > 0 else []}
        detector_wrapper = DetectorWrapper(detector_externo)

        janela_dados_recentes = []
        if len(X_init) > 0:
            num_amostras_iniciais = min(len(X_init), tamanho_janela)
            janela_dados_recentes = list(zip(X_init[:num_amostras_iniciais], y_init[:num_amostras_iniciais]))

        return scaler, modelo_atual, regime_atual_id, pools_por_regime, detector_wrapper, janela_dados_recentes

    @staticmethod
    def _processar_retreino_e_transicao_regime(
        buffer_novo_conceito, scaler, tipo_modelo, regime_atual_id,
        max_regimes_ciclo, pools_por_regime, max_pool_size
    ):
        """Retreina um modelo e gerencia a transição para um novo regime."""
        print(f"\n--- Retreinando modelo (Potencialmente para novo regime) ---")
        X_novo = np.array([x for x, y in buffer_novo_conceito])
        y_novo = np.array([y for x, y in buffer_novo_conceito])

        X_novo_scaled = scaler.transform(X_novo)

        novo_modelo_candidato = FunctionSuport.criar_modelo(tipo_modelo)
        novo_modelo_candidato.treinar(X_novo_scaled, y_novo)

        novo_regime_id = (regime_atual_id + 1) % max_regimes_ciclo
        modelo_para_ativar = novo_modelo_candidato

        if novo_regime_id not in pools_por_regime:
            pools_por_regime[novo_regime_id] = []

        pools_por_regime[novo_regime_id].append(modelo_para_ativar)
        if len(pools_por_regime[novo_regime_id]) > max_pool_size:
            pools_por_regime[novo_regime_id].pop(0)

        print(f"Novo modelo ({modelo_para_ativar.nome}) ativado para o regime {novo_regime_id}.")
        return modelo_para_ativar, novo_regime_id, pools_por_regime

    @staticmethod
    def _reagir_ao_drift_detectado(
        pools_do_regime_corrente, janela_dados_recentes, scaler, modelo_atual_original, regime_id
    ):
        """Tenta substituir o modelo atual pelo melhor do pool do regime corrente."""
        modelo_para_usar = modelo_atual_original
        if pools_do_regime_corrente:
            melhor_do_pool_regime = FunctionSuport.selecionar_melhor_modelo(
                pools_do_regime_corrente, janela_dados_recentes, scaler
            )
            if melhor_do_pool_regime:
                modelo_para_usar = melhor_do_pool_regime

        return modelo_para_usar

    @staticmethod
    def _gerenciar_adicao_periodica_ao_pool(
        pools_por_regime, regime_atual_id, modelo_atual, max_pool_size
    ):
        """Adiciona uma cópia do modelo atual ao pool do regime atual."""
        if regime_atual_id not in pools_por_regime:
            pools_por_regime[regime_atual_id] = []

        if hasattr(modelo_atual, 'nome'): # Só adiciona se o modelo_atual for válido
            novo_no_pool = copy.deepcopy(modelo_atual)
            pools_por_regime[regime_atual_id].append(novo_no_pool)
            if len(pools_por_regime[regime_atual_id]) > max_pool_size:
                pools_por_regime[regime_atual_id].pop(0)
        return pools_por_regime

    @staticmethod
    def executar_framework(X_init, y_init, X_stream, y_stream, detector_externo,
                        tipo_modelo="rf", tamanho_janela=100,
                        intervalo_adicao_pool=50, observacoes_novo_conceito=50,
                        max_pool_size=5,
                        max_regimes_ciclo=3):

        print("\n=== Inicialização do Framework com Regimes Simplificados ===")

        scaler, modelo_atual, regime_atual_id, pools_por_regime, \
        detector_wrapper, janela_dados_recentes = FrameworkDetector._inicializar_componentes(
            X_init, y_init, tipo_modelo, detector_externo, tamanho_janela
        )

        # Inicialização de contadores e buffers de estado
        contador_adicao_pool = 0
        contador_novo_conceito = 0
        coletando_dados_pos_drift = False
        buffer_novo_conceito = []

        # Listas para armazenar resultados
        predicoes = []
        erros_predicao = []
        pontos_drift = []
        indices_metricas_rmse = []
        valores_metricas_rmse = []
        tamanho_pool_historico_regime_atual = []
        modelo_ativo_historico = []
        regime_ativo_historico = []

        print("\n=== Iniciando Processamento do Stream ===")
        start_time = time.time()
        initial_size = len(X_init)

        for i, (x_t, y_t) in enumerate(tqdm(zip(X_stream, y_stream), desc="Stream", total=len(X_stream))):
            indice_global = initial_size + i

            x_t_reshaped = x_t.reshape(1, -1)
            if hasattr(scaler, 'mean_'):
                x_t_scaled = scaler.transform(x_t_reshaped)
            else:
                x_t_scaled = x_t_reshaped


            # Previsão e cálculo do erro
            try:
                y_pred = modelo_atual.prever(x_t_scaled)[0]
            except NotFittedError:
                print(f"Alerta: Modelo {modelo_atual.nome} não ajustado no índice {indice_global}. Usando y_t como predição.")
                y_pred = y_t # Isso e um falback

            predicoes.append(y_pred)
            erro = abs(y_t - y_pred)
            erros_predicao.append(erro)
            detector_wrapper.atualizar(erro)
            drift_detectado_nesta_amostra = detector_wrapper.drift_detectado

            # Rastreamento
            if not modelo_ativo_historico or modelo_ativo_historico[-1] != modelo_atual.nome:
                modelo_ativo_historico.append(modelo_atual.nome)
            regime_ativo_historico.append(regime_atual_id)

            # Lógica principal de adaptação
            if coletando_dados_pos_drift:
                contador_novo_conceito += 1
                buffer_novo_conceito.append((x_t, y_t))

                if contador_novo_conceito >= observacoes_novo_conceito:
                    modelo_atual, regime_atual_id, pools_por_regime = FrameworkDetector._processar_retreino_e_transicao_regime(
                        buffer_novo_conceito, scaler, tipo_modelo, regime_atual_id,
                        max_regimes_ciclo, pools_por_regime, max_pool_size
                    )
                    coletando_dados_pos_drift = False
                    contador_novo_conceito = 0
                    buffer_novo_conceito = []
                    contador_adicao_pool = 0
            else:
                if drift_detectado_nesta_amostra:

                    pontos_drift.append(indice_global)
                    print(f"\n!!! Drift detectado no índice {indice_global} (REGIME ATUAL: {regime_atual_id}) !!!")

                    pool_do_regime_corrente = pools_por_regime.get(regime_atual_id, [])
                    modelo_atual = FrameworkDetector._reagir_ao_drift_detectado(
                        pool_do_regime_corrente, janela_dados_recentes, scaler, modelo_atual, regime_atual_id
                    )

                    coletando_dados_pos_drift = True
                    contador_novo_conceito = 0
                    buffer_novo_conceito = [(x_t, y_t)]

                else: # Operação Normal

                    contador_adicao_pool += 1
                    if contador_adicao_pool >= intervalo_adicao_pool:
                        pools_por_regime = FrameworkDetector._gerenciar_adicao_periodica_ao_pool(
                            pools_por_regime, regime_atual_id, modelo_atual, max_pool_size
                        )
                        contador_adicao_pool = 0

            # Rastreamento do tamanho do pool e métricas
            tamanho_pool_historico_regime_atual.append(len(pools_por_regime.get(regime_atual_id, [])))

            # Esse if aqui pode parecer confuso por ter muitas condições, mas foi a melhor maneira que achei para fazer a logica que eu imagino, porem se tiver um ideia melhor, pode corrigir
            # Isso são basicamente pre-requisitos para calcular as metricas
            if i > 0 and i % 50 == 0 and len(janela_dados_recentes) > 5 and hasattr(scaler, 'mean_'):

                X_janela_metricas = np.array([x for x, _ in janela_dados_recentes])
                y_janela_metricas = np.array([y for _, y in janela_dados_recentes])
                X_janela_scaled_metricas = scaler.transform(X_janela_metricas)
                if hasattr(modelo_atual, 'prever'):
                    y_prev_janela_metricas = modelo_atual.prever(X_janela_scaled_metricas)
                    rmse = np.sqrt(mean_squared_error(y_janela_metricas, y_prev_janela_metricas))
                    indices_metricas_rmse.append(indice_global)
                    valores_metricas_rmse.append(rmse)

            janela_dados_recentes = FunctionSuport.adicionar_a_janela(
                janela_dados_recentes, (x_t, y_t), tamanho_janela
            )

        tempo_total = time.time() - start_time
        print(f"\n=== Processamento concluído em {tempo_total:.2f} segundos ===")
        print(f"Número de drifts detectados: {len(pontos_drift)}")
        print(f"Pools finais por regime: { {k: len(v) for k, v in pools_por_regime.items()} }")

        return {
            'predicoes': np.array(predicoes),
            'erros': np.array(erros_predicao),
            'pontos_drift': pontos_drift,
            'indices_rmse': indices_metricas_rmse,
            'valores_rmse': valores_metricas_rmse,
            'tamanho_pool_regime_atual': tamanho_pool_historico_regime_atual,
            'modelo_ativo': modelo_ativo_historico,
            'regime_ativo': regime_ativo_historico
        }
