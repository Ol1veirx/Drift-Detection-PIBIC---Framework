import numpy as np
import copy
import time
from tqdm.notebook import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

class StreamProcessor:
    """
    Classe que encapsula a lógica de processamento de streams com detecção de drift
    e gerenciamento inteligente de pools de modelos por regime.
    """

    def __init__(self,
                modelo_inicial,
                detector_wrapper,
                scaler,
                janela_dados_recentes,
                tipo_modelo_global,
                tamanho_janela=200,
                intervalo_adicao_pool=30,
                observacoes_novo_conceito=55,
                min_diversidade_erro=0.05,
                n_clusters_regimes=3,
                limiar_degradacao=0.15,
                threshold_melhoria_alerta=0.97,
                metrics_interval=50,
                min_samples_for_metrics=5,
                max_pool_size=5):
        """
        Inicializa o processador de stream com os parâmetros necessários.
        """
        # Modelos e detectores
        self.modelo_atual = modelo_inicial
        self.detector_wrapper = detector_wrapper
        self.scaler = scaler
        self.tipo_modelo_global = tipo_modelo_global

        # Parâmetros de configuração
        self.tamanho_janela = tamanho_janela
        self.intervalo_adicao_pool = intervalo_adicao_pool
        self.observacoes_novo_conceito = observacoes_novo_conceito
        self.min_diversidade_erro = min_diversidade_erro
        self.n_clusters_regimes = n_clusters_regimes
        self.limiar_degradacao = limiar_degradacao
        self.threshold_melhoria_alerta = threshold_melhoria_alerta
        self.metrics_interval = metrics_interval
        self.min_samples_for_metrics = min_samples_for_metrics
        self.max_pool_size = max_pool_size

        # Estado do sistema
        self.janela_dados_recentes = janela_dados_recentes
        self.modelo_a_manter_do_pool_anterior = None
        self.regime_atual = 0
        self.pools_por_regime = {0: [modelo_inicial]}
        self.contador_adicao_pool = 0
        self.drift_detectado_flag = False
        self.contador_novo_conceito = 0
        self.buffer_novo_conceito = []
        self.modelo_transicao = None

        # Métricas e resultados
        self.erros_predicao_stream = []
        self.predicoes_stream = []
        self.estados_detector_stream = []
        self.pontos_drift_detectados = []
        self.metricas_rmse_stream = []
        self.metricas_mae_stream = []
        self.metricas_r2_stream = []
        self.modelo_ativo_ao_longo_do_tempo = []
        self.tamanho_pool_ao_longo_do_tempo = []

    def _adicionar_decomposicao(self, janela_dados, n_recente=50):
        """Adiciona features baseadas em decomposição de série temporal"""
        if len(janela_dados) < n_recente:
            return None, None  # Não há dados suficientes

        # Extrai apenas os valores target da janela, garantindo que seja um array 1D simples
        y_recente = np.array([float(y) for _, y in janela_dados[-n_recente:]])

        # Verifica se o array está no formato correto antes de prosseguir
        if len(y_recente.shape) != 1:
            return None, None  # Formato inválido

        # Usa uma média móvel simples em vez de convolução para evitar problemas
        try:
            # Calcula média móvel com janela de 5 elementos
            window_size = min(5, len(y_recente))
            tendencia = np.zeros_like(y_recente)
            for i in range(len(y_recente)):
                start = max(0, i - window_size + 1)
                tendencia[i] = np.mean(y_recente[start:i+1])

            # Retorna o último valor da tendência e a inclinação
            if len(tendencia) >= 2:
                inclinacao = tendencia[-1] - tendencia[-2]
                return tendencia[-1], inclinacao
        except Exception as e:
            print(f"Aviso: Erro ao calcular decomposição: {e}")

        return None, None

    def _prever_ensemble(self, x_scaled, janela_recente, estado="NORMAL"):
        """Realiza previsão usando ensemble de modelos quando apropriado"""
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        # Em estado normal usa apenas o modelo atual
        if estado == "NORMAL":
            return self.modelo_atual.prever(x_scaled)[0]

        # Durante ALERTA ou MUDANÇA, usa ensemble dos melhores modelos
        pool_atual = self.pools_por_regime.get(self.regime_atual, [])
        if len(pool_atual) <= 1:
            return self.modelo_atual.prever(x_scaled)[0]

        # Seleciona os top 3 modelos do pool (ou menos se não houver 3)
        erros_modelos = []
        for modelo in pool_atual:
            erro = FrameworkDetector.desempenho(modelo, janela_recente, self.scaler)
            erros_modelos.append((modelo, erro))

        # Ordena por erro (menor primeiro)
        erros_modelos.sort(key=lambda x: x[1])
        top_modelos = erros_modelos[:min(3, len(erros_modelos))]

        # Pesos baseados no inverso do erro (quanto menor o erro, maior o peso)
        pesos = [1/(erro+0.001) for _, erro in top_modelos]
        soma_pesos = sum(pesos)
        pesos_normalizados = [p/soma_pesos for p in pesos]

        # Previsão ponderada
        predicao = 0
        for (modelo, _), peso in zip(top_modelos, pesos_normalizados):
            predicao += modelo.prever(x_scaled)[0] * peso

        return predicao

    def _criar_modelo_transicao(self, janela_recente):
        """Cria um modelo simples e rápido para período de transição"""
        if len(janela_recente) < 30:
            return None

        X_janela = np.array([x for x, _ in janela_recente[-30:]])  # últimas 30 amostras
        y_janela = np.array([y for _, y in janela_recente[-30:]])

        modelo_transicao = LinearRegression()  # Modelo simples e rápido
        modelo_transicao.fit(self.scaler.transform(X_janela), y_janela)

        # Wrapper para compatibilidade com a interface
        class ModeloTransicao:
            def __init__(self, modelo):
                self.modelo = modelo
                self.nome = "ModeloTransicao"

            def prever(self, X):
                return self.modelo.predict(X)

        return ModeloTransicao(modelo_transicao)

    def _transicao_suave(self, modelo_antigo, modelo_novo, x_t_scaled, pct_transicao):
        """Faz transição suave entre modelos usando interpolação"""
        pred_antigo = modelo_antigo.prever(x_t_scaled)[0]
        pred_novo = modelo_novo.prever(x_t_scaled)[0]

        # Quanto maior o pct_transicao, mais peso ao modelo novo
        return pred_antigo * (1 - pct_transicao) + pred_novo * pct_transicao

    def processar_stream(self, X_stream, Y_stream, initial_size, detector_escolhido):
        """
        Processa o stream de dados, detectando drifts e adaptando modelos.
        """
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        print("\n=== Iniciando Processamento do Stream ===")
        print(f"Processando {len(X_stream)} amostras...")

        start_time = time.time()

        # Loop principal de processamento do stream
        for i, (x_t, y_t) in enumerate(tqdm(zip(X_stream, Y_stream), total=len(X_stream), desc="Stream")):
            indice_global = initial_size + i

            # --- 1. Predição com o modelo atual ---
            x_t_reshaped = x_t.reshape(1, -1)
            x_t_scaled = self.scaler.transform(x_t_reshaped)

            # Obter estado do detector antes de usar para previsão
            estado = FrameworkDetector.get_state(self.detector_wrapper)
            self.estados_detector_stream.append(estado)

            # Adicionando tendência para melhorar previsão
            tendencia, inclinacao = self._adicionar_decomposicao(self.janela_dados_recentes)

            # Previsão adaptativa baseada no estado
            if self.drift_detectado_flag:
                # Durante coleta pós-drift, usa modelo de transição ou ensemble
                if not hasattr(self, 'modelo_transicao') or self.modelo_transicao is None or i % 10 == 0:  # atualiza a cada 10 passos
                    self.modelo_transicao = self._criar_modelo_transicao(self.janela_dados_recentes)

                # Se temos modelo de transição, use-o
                if self.modelo_transicao:
                    # Interpola entre modelo atual e de transição
                    pct_transicao = min(self.contador_novo_conceito / self.observacoes_novo_conceito, 0.7)
                    y_pred = self._transicao_suave(self.modelo_atual, self.modelo_transicao, x_t_scaled, pct_transicao)
                else:
                    # Se não foi possível criar modelo de transição, use o atual
                    y_pred = self.modelo_atual.prever(x_t_scaled)[0]
            else:
                # Em operação normal, usa modelo atual ou ensemble conforme estado
                y_pred = self._prever_ensemble(x_t_scaled, self.janela_dados_recentes, estado)

            # Aplica ajuste de tendência se disponível
            if tendencia is not None and abs(tendencia) > 0.1:
                y_pred = y_pred * 0.9 + tendencia * 0.1  # Suave incorporação da tendência

            self.predicoes_stream.append(y_pred)

            # --- 2. Cálculo do Erro e Atualização do Detector ---
            erro = abs(y_t - y_pred)
            valor_ou_erro_para_detector = y_t if detector_escolhido == "KSWIN" else erro
            self.detector_wrapper.atualizar(valor_ou_erro_para_detector)
            self.erros_predicao_stream.append(erro)

            # --- 3. Identificar Regime ---
            self._identificar_regime(i)

            # --- 4. Lógica de Adaptação ---
            if self.drift_detectado_flag:
                self._processar_estado_drift(x_t, y_t, x_t_scaled)
            else:
                self._processar_estado_normal(estado, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido)

            # --- 5. Armazenamento de Resultados ---
            self.modelo_ativo_ao_longo_do_tempo.append(
                self.modelo_atual.nome if hasattr(self.modelo_atual, 'nome') else type(self.modelo_atual).__name__
            )
            self.tamanho_pool_ao_longo_do_tempo.append(
                sum(len(pool) for pool in self.pools_por_regime.values())
            )

            # --- 6. Cálculo Periódico de Métricas ---
            self._calcular_metricas_periodicas(i, indice_global, estado)

            # --- 7. Atualização da Janela de Dados Recentes ---
            self.janela_dados_recentes = FrameworkDetector.adicionar_a_janela(
                self.janela_dados_recentes, (x_t, y_t), tamanho_max=self.tamanho_janela
            )

        # --- Relatório Final ---
        end_time = time.time()
        print(f"\n=== Processamento Concluído em {end_time - start_time:.2f} segundos ===")
        print(f"Drifts detectados: {len(self.pontos_drift_detectados)}")
        if self.pontos_drift_detectados:
            print(f"  Índices: {self.pontos_drift_detectados}")

        # Sumário dos pools por regime
        print("\n--- Pools por Regime ---")
        total_modelos = 0
        for regime, pool in self.pools_por_regime.items():
            print(f"Regime {regime}: {len(pool)} modelo(s)")
            for i, modelo in enumerate(pool):
                print(f"  {i+1}. {modelo.nome if hasattr(modelo, 'nome') else type(modelo).__name__}")
            total_modelos += len(pool)
        print(f"Total de modelos: {total_modelos}")

        # Retornando os resultados para análise posterior
        return {
            'erros_predicao_stream': self.erros_predicao_stream,
            'predicoes_stream': self.predicoes_stream,
            'estados_detector_stream': self.estados_detector_stream,
            'pontos_drift_detectados': self.pontos_drift_detectados,
            'metricas_rmse_stream': self.metricas_rmse_stream,
            'metricas_mae_stream': self.metricas_mae_stream,
            'metricas_r2_stream': self.metricas_r2_stream,
            'modelo_ativo_ao_longo_do_tempo': self.modelo_ativo_ao_longo_do_tempo,
            'tamanho_pool_ao_longo_do_tempo': self.tamanho_pool_ao_longo_do_tempo,
            'pools_por_regime': self.pools_por_regime
        }

    def _identificar_regime(self, i):
        """Identifica regime e gerencia pools de regimes"""
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        if i % 20 == 0 and len(self.janela_dados_recentes) >= self.min_samples_for_metrics:
            regime_anterior = self.regime_atual
            self.regime_atual = FrameworkDetector.identificar_regime(
                self.janela_dados_recentes,
                n_clusters=self.n_clusters_regimes
            )

            # Gerenciar pools por regime
            if self.regime_atual not in self.pools_por_regime:
                # Inicializa novo pool para este regime
                modelo_copy = copy.deepcopy(self.modelo_atual)
                if hasattr(modelo_copy, 'nome'):
                    modelo_copy.nome = f"{modelo_copy.nome}_R{self.regime_atual}"
                self.pools_por_regime[self.regime_atual] = [modelo_copy]
                print(f"\n  🆕 Novo regime {self.regime_atual} detectado com modelo: {modelo_copy.nome if hasattr(modelo_copy, 'nome') else type(modelo_copy).__name__}")
            elif self.regime_atual != regime_anterior:
                print(f"\n  ⚠️ Mudança de regime: {regime_anterior} → {self.regime_atual}")

    def _processar_estado_drift(self, x_t, y_t, x_t_scaled):
        """Processa o estado após detecção de drift (coleta e retreino)"""
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        # --- COLETA DE DADOS APÓS DRIFT DETECTADO ---
        self.contador_novo_conceito += 1
        self.buffer_novo_conceito.append((x_t, y_t))

        # NOVA FUNCIONALIDADE: Continue adaptando o modelo atual durante a coleta
        if hasattr(self.modelo_atual, "partial_fit"):
            self.modelo_atual.partial_fit(x_t_scaled, np.array([y_t]))

        if self.contador_novo_conceito >= self.observacoes_novo_conceito:
            # Retreinamento após coletar dados suficientes
            print(f"\n--- Retreinando Modelo com {self.observacoes_novo_conceito} obs. do novo conceito ---")
            X_novo_list = np.array([x for x, y in self.buffer_novo_conceito])
            y_novo_array = np.array([y for x, y in self.buffer_novo_conceito])

            novo_modelo = FrameworkDetector.treinar_novo_conceito(
                X_novo_list, y_novo_array, self.scaler, tipo_modelo=self.tipo_modelo_global
            )

            if novo_modelo:
                # Configurar o novo modelo
                if hasattr(novo_modelo, 'nome'):
                    novo_modelo.nome = f"{self.tipo_modelo_global.__name__}_R{self.regime_atual}_D{len(self.pontos_drift_detectados)}"

                self.modelo_atual = novo_modelo
                self.pools_por_regime[self.regime_atual] = [self.modelo_atual]  # Reset do pool

                # Adicionar modelo anterior se for diverso
                if self.modelo_a_manter_do_pool_anterior and self.modelo_a_manter_do_pool_anterior is not self.modelo_atual:
                    erro_atual = FrameworkDetector.desempenho(self.modelo_atual, self.janela_dados_recentes, self.scaler)
                    erro_anterior = FrameworkDetector.desempenho(self.modelo_a_manter_do_pool_anterior, self.janela_dados_recentes, self.scaler)

                    if abs(erro_atual - erro_anterior) >= self.min_diversidade_erro:
                        self.pools_por_regime[self.regime_atual].append(self.modelo_a_manter_do_pool_anterior)
                        print(f"  ✓ Modelo anterior mantido no pool (diverso)")
                    else:
                        print(f"  ⚠️ Modelo anterior descartado (não diverso)")

                print(f"  ✓ Novo modelo ativo: '{self.modelo_atual.nome if hasattr(self.modelo_atual, 'nome') else type(self.modelo_atual).__name__}'")
            else:
                print(f"  ⚠️ Falha no retreino. Mantendo modelo atual.")

            # Reset dos flags e contadores
            self.drift_detectado_flag = False
            self.contador_novo_conceito = 0
            self.buffer_novo_conceito = []
            self.contador_adicao_pool = 0
            self.modelo_transicao = None  # Limpa o modelo de transição
            print("--- Retomando operação normal ---")

    def _processar_estado_normal(self, estado, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido):
        """Processa os estados normais (NORMAL, ALERTA, MUDANÇA)"""
        if estado == "MUDANÇA":
            self._processar_mudanca(indice_global, x_t, y_t, detector_escolhido)
        elif estado == "ALERTA":
            self._processar_alerta(indice_global, x_t_scaled, y_t, i)
        elif estado == "NORMAL":
            self._processar_normal(x_t_scaled, y_t)

    def _processar_mudanca(self, indice_global, x_t, y_t, detector_escolhido):
        """Processa o estado de MUDANÇA (drift detectado)"""
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        # Drift detectado - preparação para adaptação
        self.pontos_drift_detectados.append(indice_global)
        print(f"\n!!! Drift detectado no índice {indice_global} (Detector: {detector_escolhido}, Regime: {self.regime_atual}) !!!")

        # Selecionar pool apropriado para o regime atual
        pool_atual = self.pools_por_regime.get(self.regime_atual, [])
        if not pool_atual:
            pool_atual = self.pools_por_regime.get(0, [])  # Fallback para regime padrão
            print(f"  ⚠️ Usando pool do regime 0 como fallback")

        # Avaliação de modelos no pool
        print(f"  Avaliando {len(pool_atual)} modelos do pool...")
        melhor_do_pool = FrameworkDetector.selecionar_melhor_modelo(pool_atual, self.janela_dados_recentes, self.scaler)

        if melhor_do_pool:
            # Comparação com modelo atual
            erro_atual = FrameworkDetector.desempenho(self.modelo_atual, self.janela_dados_recentes, self.scaler)
            erro_melhor = FrameworkDetector.desempenho(melhor_do_pool, self.janela_dados_recentes, self.scaler)
            print(f"  Desempenho: atual={erro_atual:.4f}, melhor do pool={erro_melhor:.4f}")

            if erro_melhor < erro_atual:
                self.modelo_atual = melhor_do_pool
                print(f"  ✓ Novo modelo ativo: '{self.modelo_atual.nome if hasattr(self.modelo_atual, 'nome') else type(self.modelo_atual).__name__}'")

            # Guarda cópia do modelo atual como reserva
            self.modelo_a_manter_do_pool_anterior = copy.deepcopy(self.modelo_atual)
        else:
            self.modelo_a_manter_do_pool_anterior = copy.deepcopy(self.modelo_atual)
            print("  ⚠️ Mantendo modelo atual (pool sem alternativas)")

        # Reorganiza o pool após drift (mantém modelos bons, remove piores)
        self.pools_por_regime[self.regime_atual] = FrameworkDetector.gerenciar_pool(
            pool_atual, self.janela_dados_recentes, self.scaler,
            estado_detector="MUDANÇA", modelo_atual=self.modelo_atual, max_pool_size=self.max_pool_size
        )

        # Inicia coleta para novo conceito
        self.drift_detectado_flag = True
        self.buffer_novo_conceito = [(x_t, y_t)]

        # Cria modelo de transição logo quando detecta drift
        self.modelo_transicao = self._criar_modelo_transicao(self.janela_dados_recentes)

        print(f"  Iniciando coleta para retreino ({self.observacoes_novo_conceito} amostras)")

    def _processar_alerta(self, indice_global, x_t_scaled, y_t, i):
        """Processa o estado de ALERTA"""
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        # Estado de alerta - busca melhor modelo no pool atual
        pool_atual = self.pools_por_regime.get(self.regime_atual, [])

        if len(pool_atual) > 1:
            melhor_do_pool = FrameworkDetector.selecionar_melhor_modelo(pool_atual, self.janela_dados_recentes, self.scaler)

            if melhor_do_pool and melhor_do_pool is not self.modelo_atual:
                erro_atual = FrameworkDetector.desempenho(self.modelo_atual, self.janela_dados_recentes, self.scaler)
                erro_melhor = FrameworkDetector.desempenho(melhor_do_pool, self.janela_dados_recentes, self.scaler)

                if erro_melhor < erro_atual * self.threshold_melhoria_alerta:
                    self.modelo_atual = melhor_do_pool
                    print(f"\n  🔄 Troca durante alerta: '{self.modelo_atual.nome if hasattr(self.modelo_atual, 'nome') else type(self.modelo_atual).__name__}' (erro: {erro_melhor:.4f})")

        # Adaptação incremental se disponível
        if hasattr(self.modelo_atual, "partial_fit"):
            self.modelo_atual.partial_fit(x_t_scaled, np.array([y_t]))

        # Diversificação do pool durante alerta
        if i % 10 == 0 and hasattr(self.modelo_atual, "criar_variante"):
            variante = self.modelo_atual.criar_variante(mutation_strength=0.1)
            if variante:
                self.pools_por_regime[self.regime_atual] = FrameworkDetector.adicionar_ao_pool(
                    self.pools_por_regime.get(self.regime_atual, []), variante, self.max_pool_size,
                    self.janela_dados_recentes, self.scaler, self.min_diversidade_erro
                )

    def _processar_normal(self, x_t_scaled, y_t):
        """Processa o estado NORMAL"""
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        # Estado normal - manutenção periódica do pool
        self.contador_adicao_pool += 1
        if self.contador_adicao_pool >= self.intervalo_adicao_pool:
            # Adicionar cópia do modelo atual ao pool
            novo_modelo_pool = copy.deepcopy(self.modelo_atual)
            if hasattr(novo_modelo_pool, 'nome'):
                novo_modelo_pool.nome = f"{novo_modelo_pool.nome}_Copy{self.contador_adicao_pool}"

            # Gerenciar adição ao pool do regime atual
            pool_atual = self.pools_por_regime.get(self.regime_atual, [])
            if not pool_atual:
                self.pools_por_regime[self.regime_atual] = [novo_modelo_pool]
            else:
                self.pools_por_regime[self.regime_atual] = FrameworkDetector.adicionar_ao_pool(
                    pool_atual, novo_modelo_pool, self.max_pool_size,
                    self.janela_dados_recentes, self.scaler, self.min_diversidade_erro
                )

            self.contador_adicao_pool = 0

    def _calcular_metricas_periodicas(self, i, indice_global, estado):
        """Calcula métricas periodicamente e detecta degradação"""
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        if i > 0 and i % self.metrics_interval == 0 and len(self.janela_dados_recentes) >= self.min_samples_for_metrics:
            X_janela_eval = np.array([x for x, _ in self.janela_dados_recentes])
            y_janela_eval = np.array([y for _, y in self.janela_dados_recentes])
            X_janela_eval_scaled = self.scaler.transform(X_janela_eval)
            y_prev_eval = self.modelo_atual.prever(X_janela_eval_scaled)

            # Cálculo de métricas
            rmse = np.sqrt(mean_squared_error(y_janela_eval, y_prev_eval))
            mae = mean_absolute_error(y_janela_eval, y_prev_eval)
            r2 = r2_score(y_janela_eval, y_prev_eval)

            # Armazenamento de métricas
            self.metricas_rmse_stream.append((indice_global, rmse))
            self.metricas_mae_stream.append((indice_global, mae))
            self.metricas_r2_stream.append((indice_global, r2))

            # Detecção de degradação de desempenho
            if mae > self.limiar_degradacao and not self.drift_detectado_flag and estado != "MUDANÇA":
                print(f"\n  ⚠️ Degradação detectada (MAE: {mae:.4f})")

                # Busca em todos os pools por um modelo melhor
                melhor_modelo, melhor_erro, melhor_regime = None, float('inf'), None

                for reg, pool in self.pools_por_regime.items():
                    if pool:
                        melhor_do_reg = FrameworkDetector.selecionar_melhor_modelo(pool, self.janela_dados_recentes, self.scaler)
                        if melhor_do_reg:
                            erro = FrameworkDetector.desempenho(melhor_do_reg, self.janela_dados_recentes, self.scaler)
                            if erro < melhor_erro:
                                melhor_erro = erro
                                melhor_modelo = melhor_do_reg
                                melhor_regime = reg

                # Troca para o melhor modelo se for significativamente melhor
                if melhor_modelo and melhor_erro < mean_squared_error(y_janela_eval, y_prev_eval) * 0.9:
                    self.modelo_atual = melhor_modelo
                    print(f"  🔄 Novo modelo do regime {melhor_regime}: '{melhor_modelo.nome if hasattr(melhor_modelo, 'nome') else type(melhor_modelo).__name__}'")
                    print(f"  📊 Erro esperado: {melhor_erro:.4f}")