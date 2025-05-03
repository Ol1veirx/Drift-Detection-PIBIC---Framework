import copy
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.notebook import tqdm

from frameworkClasses.alert_state_processor import AlertStateProcessor
from frameworkClasses.change_state_processor import ChangeStateProcessor
from frameworkClasses.drift_collection_processor import DriftCollectionProcessor
from frameworkClasses.drift_prediction_strategy import DriftPredictionStrategy
from frameworkClasses.normal_state_processor import NormalStateProcessor
from frameworkClasses.standard_prediction_strategy import StandardPredictionStrategy
from regressores.modelosOffline.LinearRegressionModelo import LinearRegressionModelo
from regressores.modelosOffline import LinearRegressionModelo


class StreamProcessor:
    """Processa streams com detecção de drift e gerenciamento inteligente de pools de modelos por regime"""

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

        # Inicialização de atributos
        self.modelo_atual = modelo_inicial
        self.detector_wrapper = detector_wrapper
        self.scaler = scaler
        self.tipo_modelo_global = tipo_modelo_global
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
        self.janela_dados_recentes = janela_dados_recentes
        self.modelo_a_manter_do_pool_anterior = None
        self.regime_atual = 0
        self.pools_por_regime = {0: [modelo_inicial]}
        self.contador_adicao_pool = 0
        self.drift_detectado_flag = False
        self.contador_novo_conceito = 0
        self.buffer_novo_conceito = []
        self.modelo_transicao = None
        self.erros_predicao_stream = []
        self.predicoes_stream = []
        self.estados_detector_stream = []
        self.pontos_drift_detectados = []
        self.metricas_rmse_stream = []
        self.metricas_mae_stream = []
        self.metricas_r2_stream = []
        self.modelo_ativo_ao_longo_do_tempo = []
        self.tamanho_pool_ao_longo_do_tempo = []

        # Estratégias
        self.prediction_strategies = {
            "standard": StandardPredictionStrategy(),
            "drift": DriftPredictionStrategy()
        }

        self.state_processors = {
            "NORMAL": NormalStateProcessor(),
            "ALERTA": AlertStateProcessor(),
            "MUDANÇA": ChangeStateProcessor(),
            "DRIFT_COLLECTION": DriftCollectionProcessor()
        }

    def _adicionar_decomposicao(self, janela_dados=None, n_recente=50):
        """Extrai tendência e inclinação da série temporal recente"""
        if janela_dados is None:
            janela_dados = self.janela_dados_recentes

        if len(janela_dados) < n_recente:
            return None, None

        try:
            y_recente = np.array([float(y) for _, y in janela_dados[-n_recente:]])

            if len(y_recente.shape) != 1:
                return None, None

            window_size = min(5, len(y_recente))
            tendencia = np.zeros_like(y_recente)
            for i in range(len(y_recente)):
                start = max(0, i - window_size + 1)
                tendencia[i] = np.mean(y_recente[start:i+1])

            if len(tendencia) >= 2:
                inclinacao = tendencia[-1] - tendencia[-2]
                return tendencia[-1], inclinacao
        except Exception:
            pass

        return None, None

    def _criar_modelo_transicao(self):
        """Cria um modelo linear simples para uso durante períodos de transição"""
        if len(self.janela_dados_recentes) < 30:
            return None

        try:
            X_janela = np.array([x for x, _ in self.janela_dados_recentes[-30:]])
            y_janela = np.array([y for _, y in self.janela_dados_recentes[-30:]])

            modelo_transicao = LinearRegressionModelo()
            modelo_transicao.treinar(self.scaler.transform(X_janela), y_janela)

            class ModeloTransicao:
                def __init__(self, modelo):
                    self.modelo = modelo
                    self.nome = "ModeloTransicao"

                def prever(self, X):
                    return self.modelo.prever(X)

            return ModeloTransicao(modelo_transicao)
        except Exception:
            return None

    def _identificar_regime(self, i):
        """Identifica e gerencia regimes nos dados"""
        from frameworkDetector.framework_detector import FrameworkDetector

        if i % 20 == 0 and len(self.janela_dados_recentes) >= self.min_samples_for_metrics:
            regime_anterior = self.regime_atual
            self.regime_atual = FrameworkDetector.identificar_regime(
                self.janela_dados_recentes,
                n_clusters=self.n_clusters_regimes
            )

            if self.regime_atual not in self.pools_por_regime:
                modelo_copy = copy.deepcopy(self.modelo_atual)
                if hasattr(modelo_copy, 'nome'):
                    modelo_copy.nome = f"{modelo_copy.nome}_R{self.regime_atual}"
                self.pools_por_regime[self.regime_atual] = [modelo_copy]
                print(f"\n  🆕 Novo regime {self.regime_atual} detectado com modelo: {modelo_copy.nome if hasattr(modelo_copy, 'nome') else type(modelo_copy).__name__}")
            elif self.regime_atual != regime_anterior:
                print(f"\n  ⚠️ Mudança de regime: {regime_anterior} → {self.regime_atual}")

    def _calcular_metricas_periodicas(self, i, indice_global, estado):
        """Calcula métricas e detecta degradação de desempenho periodicamente"""
        from frameworkDetector.framework_detector import FrameworkDetector

        if i > 0 and i % self.metrics_interval == 0 and len(self.janela_dados_recentes) >= self.min_samples_for_metrics:
            X_janela_eval = np.array([x for x, _ in self.janela_dados_recentes])
            y_janela_eval = np.array([y for _, y in self.janela_dados_recentes])
            X_janela_eval_scaled = self.scaler.transform(X_janela_eval)
            y_prev_eval = self.modelo_atual.prever(X_janela_eval_scaled)

            rmse = np.sqrt(mean_squared_error(y_janela_eval, y_prev_eval))
            mae = mean_absolute_error(y_janela_eval, y_prev_eval)
            r2 = r2_score(y_janela_eval, y_prev_eval)

            self.metricas_rmse_stream.append((indice_global, rmse))
            self.metricas_mae_stream.append((indice_global, mae))
            self.metricas_r2_stream.append((indice_global, r2))

            if mae > self.limiar_degradacao and not self.drift_detectado_flag and estado != "MUDANÇA":
                print(f"\n  ⚠️ Degradação detectada (MAE: {mae:.4f})")

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

                if melhor_modelo and melhor_erro < mean_squared_error(y_janela_eval, y_prev_eval) * 0.9:
                    self.modelo_atual = melhor_modelo
                    print(f"  🔄 Novo modelo do regime {melhor_regime}: '{melhor_modelo.nome if hasattr(melhor_modelo, 'nome') else type(melhor_modelo).__name__}'")
                    print(f"  📊 Erro esperado: {melhor_erro:.4f}")

    def processar_stream(self, X_stream, Y_stream, initial_size, detector_escolhido):
        """Processa o stream de dados, detectando drifts e adaptando modelos"""
        from frameworkDetector.framework_detector import FrameworkDetector

        print("\n=== Iniciando Processamento do Stream ===")
        print(f"Processando {len(X_stream)} amostras...")

        start_time = time.time()

        for i, (x_t, y_t) in enumerate(tqdm(zip(X_stream, Y_stream), total=len(X_stream), desc="Stream")):
            indice_global = initial_size + i

            # Preparação da entrada
            x_t_reshaped = x_t.reshape(1, -1)
            x_t_scaled = self.scaler.transform(x_t_reshaped)

            # Obter estado do detector
            estado = FrameworkDetector.get_state(self.detector_wrapper)
            self.estados_detector_stream.append(estado)

            # Adiciona decomposição
            tendencia, inclinacao = self._adicionar_decomposicao()

            # Seleciona estratégia de previsão apropriada
            if self.drift_detectado_flag:
                predicao = self.prediction_strategies["drift"].predict(self, x_t_scaled, i)
            else:
                predicao = self.prediction_strategies["standard"].predict(self, x_t_scaled, estado)

            # Aplica ajuste de tendência
            if tendencia is not None and abs(tendencia) > 0.1:
                predicao = predicao * 0.9 + tendencia * 0.1

            self.predicoes_stream.append(predicao)

            # Cálculo do erro e atualização do detector
            erro = abs(y_t - predicao)
            valor_para_detector = y_t if detector_escolhido == "KSWIN" else erro
            self.detector_wrapper.atualizar(valor_para_detector)
            self.erros_predicao_stream.append(erro)

            # Identificação de regime
            self._identificar_regime(i)

            # Processamento do estado
            if self.drift_detectado_flag:
                self.state_processors["DRIFT_COLLECTION"].process(
                    self, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido)
            else:
                # Seleciona o processador apropriado para o estado atual
                processor = self.state_processors.get(estado, self.state_processors["NORMAL"])
                processor.process(self, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido)

            # Armazenamento de resultados
            self.modelo_ativo_ao_longo_do_tempo.append(
                self.modelo_atual.nome if hasattr(self.modelo_atual, 'nome') else type(self.modelo_atual).__name__
            )
            self.tamanho_pool_ao_longo_do_tempo.append(
                sum(len(pool) for pool in self.pools_por_regime.values())
            )

            # Cálculo periódico de métricas
            self._calcular_metricas_periodicas(i, indice_global, estado)

            # Atualização da janela de dados recentes
            self.janela_dados_recentes = FrameworkDetector.adicionar_a_janela(
                self.janela_dados_recentes, (x_t, y_t), tamanho_max=self.tamanho_janela
            )

        # Relatório final
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

        # Resultados para análise
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