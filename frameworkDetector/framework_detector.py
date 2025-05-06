import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import traceback
import copy
from sklearn.cluster import KMeans
from tqdm import tqdm

# Importar wrappers de modelo
from frameworkDetector.detector_wrapper import DetectorWrapper
from frameworkDetector.suport_functions import FunctionSuport
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

    def executar_framework(X_init, y_init, X_stream, y_stream, detector_externo,
                        tipo_modelo="rf", tamanho_janela=100,
                        intervalo_adicao_pool=50, observacoes_novo_conceito=50,
                        max_pool_size=5):
        """
        Executa o framework de detecção de drift e adaptação.

        Parâmetros:
        -----------
        X_init, y_init : arrays
            Dados iniciais para treinar o modelo base
        X_stream, y_stream : arrays
            Dados do stream para processamento
        detector_externo : objeto
            Detector de drift previamente instanciado
        tipo_modelo : str
            Tipo de modelo a ser usado ('rf', 'sgd', 'knn')
        tamanho_janela : int
            Tamanho da janela deslizante para avaliação
        intervalo_adicao_pool : int
            Passo 1: Adicionar modelo ao pool a cada X observações
        observacoes_novo_conceito : int
            Passo 4: Número de obs. para retreinar após drift
        max_pool_size : int
            Tamanho máximo do pool de modelos

        Retorna:
        --------
        dict : Resultados e métricas do processamento
        """
        print("\n=== Inicialização do Framework ===")
        # Preparar scaler com dados iniciais
        scaler = StandardScaler()
        X_init_scaled = scaler.fit_transform(X_init)

        # Treinar modelo inicial
        modelo_inicial = FunctionSuport.criar_modelo(tipo_modelo)
        modelo_inicial.treinar(X_init_scaled, y_init)
        print(f"Modelo inicial ({modelo_inicial.nome}) treinado com {len(X_init)} amostras.")

        # Inicializar variáveis
        modelo_atual = modelo_inicial
        pool_modelos = [modelo_inicial]

        # Envolver o detector externo no wrapper
        detector_wrapper = DetectorWrapper(detector_externo)

        # Inicializar janela e dados de rastreamento
        janela_dados_recentes = []
        if len(X_init) > 0:
            indices_iniciais = list(range(min(len(X_init), tamanho_janela)))
            janela_dados_recentes = list(zip(X_init[indices_iniciais], y_init[indices_iniciais]))

        # Variáveis para controle do fluxo
        contador_adicao_pool = 0
        contador_novo_conceito = 0
        drift_detectado_flag = False
        buffer_novo_conceito = []

        # Estruturas para armazenar resultados
        erros_predicao = []
        pontos_drift = []
        predicoes = []
        indices_metricas_rmse = []
        valores_metricas_rmse = []
        tamanho_pool_historico = []
        modelo_ativo_historico = []

        print("\n=== Iniciando Processamento do Stream ===")
        start_time = time.time()
        initial_size = len(X_init)

        # Loop principal sobre os dados do stream
        for i, (x_t, y_t) in enumerate(tqdm(zip(X_stream, y_stream), desc="Stream", total=len(X_stream))):
            indice_global = initial_size + i

            # --- 1. Predição ---
            x_t_reshaped = x_t.reshape(1, -1)
            x_t_scaled = scaler.transform(x_t_reshaped)
            y_pred = modelo_atual.prever(x_t_scaled)[0]
            predicoes.append(y_pred)

            # --- 2. Calcular erro e atualizar detector ---
            erro = abs(y_t - y_pred)
            detector_wrapper.atualizar(erro)
            erros_predicao.append(erro)

            # --- 3. Verificar estado do detector ---
            drift_detectado = detector_wrapper.drift_detectado

            # --- 4. Ações com base no estado ---

            # --- PASSO 4: Coletando dados para retreino após drift ---
            if drift_detectado_flag:
                contador_novo_conceito += 1
                buffer_novo_conceito.append((x_t, y_t))

                # Treino incremental durante a coleta, se suportado
                if modelo_atual.suporta_incremental:
                    modelo_atual.partial_fit(x_t_scaled, np.array([y_t]))

                # Verificar se coletou dados suficientes para retreino completo
                if contador_novo_conceito >= observacoes_novo_conceito:
                    print(f"\n--- Retreinando modelo após {observacoes_novo_conceito} observações do novo conceito ---")
                    # Preparar dados coletados
                    X_novo = np.array([x for x, y in buffer_novo_conceito])
                    y_novo = np.array([y for x, y in buffer_novo_conceito])
                    X_novo_scaled = scaler.transform(X_novo)

                    # Criar e treinar novo modelo
                    novo_modelo = FunctionSuport.criar_modelo(tipo_modelo)
                    sucesso = novo_modelo.treinar(X_novo_scaled, y_novo)

                    if sucesso:
                        modelo_atual = novo_modelo
                        pool_modelos.append(modelo_atual)  # Adicionar ao pool

                        # Poda se pool excede tamanho máximo (remove mais antigo)
                        if len(pool_modelos) > max_pool_size:
                            modelo_removido = pool_modelos.pop(0)

                        print(f"  ✓ Novo modelo ({modelo_atual.nome}) ativado. Pool agora com {len(pool_modelos)} modelos.")
                    else:
                        print("  ⚠️ Falha ao treinar novo modelo. Mantendo o atual.")

                    # Reset de flags e contadores
                    drift_detectado_flag = False
                    contador_novo_conceito = 0
                    buffer_novo_conceito = []
                    contador_adicao_pool = 0

            # Operação normal / detecção de drift
            else:
                # --- PASSO 3: Drift detectado ---
                if drift_detectado:
                    pontos_drift.append(indice_global)
                    print(f"\n!!! Drift detectado no índice {indice_global} !!!")

                    # Selecionar melhor modelo do pool para a janela atual
                    melhor_do_pool = FunctionSuport.selecionar_melhor_modelo(pool_modelos, janela_dados_recentes, scaler)
                    if melhor_do_pool:
                        modelo_atual = melhor_do_pool
                        print(f"  ✓ Modelo substituído pelo melhor do pool: {modelo_atual.nome}")
                    else:
                        print("  ⚠️ Não foi possível selecionar modelo melhor. Mantendo atual.")

                    # Iniciar coleta para retreino
                    drift_detectado_flag = True
                    contador_novo_conceito = 0
                    buffer_novo_conceito = [(x_t, y_t)]

                # --- PASSO 1: Estado normal - adição periódica ao pool ---
                else:  # Normal
                    contador_adicao_pool += 1
                    # Adicionar ao pool periodicamente
                    if contador_adicao_pool >= intervalo_adicao_pool:
                        novo_no_pool = copy.deepcopy(modelo_atual)
                        pool_modelos.append(novo_no_pool)

                        # Poda se pool excede tamanho máximo
                        if len(pool_modelos) > max_pool_size:
                            modelo_removido = pool_modelos.pop(0)  # Remove o mais antigo

                        contador_adicao_pool = 0

            # --- 5. Rastreamento e métricas ---
            modelo_ativo_historico.append(modelo_atual.nome)
            tamanho_pool_historico.append(len(pool_modelos))

            # Calcular métricas periodicamente (a cada 50 amostras)
            if i % 50 == 0 and len(janela_dados_recentes) > 5:
                X_janela = np.array([x for x, _ in janela_dados_recentes])
                y_janela = np.array([y for _, y in janela_dados_recentes])
                X_janela_scaled = scaler.transform(X_janela)
                y_prev_janela = modelo_atual.prever(X_janela_scaled)

                rmse = np.sqrt(mean_squared_error(y_janela, y_prev_janela))
                indices_metricas_rmse.append(indice_global)
                valores_metricas_rmse.append(rmse)

            # --- 6. Atualizar janela de dados recentes ---
            janela_dados_recentes = FunctionSuport.adicionar_a_janela(janela_dados_recentes, (x_t, y_t), tamanho_janela)

        # --- Fim do processamento ---
        tempo_total = time.time() - start_time
        print(f"\n=== Processamento concluído em {tempo_total:.2f} segundos ===")
        print(f"Número de drifts detectados: {len(pontos_drift)}")

        # Retornar resultados
        return {
            'predicoes': np.array(predicoes),
            'erros': np.array(erros_predicao),
            'pontos_drift': pontos_drift,
            'indices_rmse': indices_metricas_rmse,
            'valores_rmse': valores_metricas_rmse,
            'tamanho_pool': tamanho_pool_historico,
            'modelo_ativo': modelo_ativo_historico
        }