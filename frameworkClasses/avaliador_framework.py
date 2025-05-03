import time
from sklearn.metrics import mean_absolute_error

from avaliacao.AvaliadorDriftBase import AvaliadorBatch
from detectores.KSWINDetector import KSWINDetector
from frameworkDetector.framework_detector import FrameworkDetector
from regressores.modelosOffline.RandomForestModelo import RandomForestModelo
from utils.StreamProcessor import StreamProcessor


class AvaliadorFramework(AvaliadorBatch):
    def __init__(self, framework_params):
        """
        framework_params: Dicionário com parâmetros para inicializar o StreamProcessor
        """
        super().__init__()
        self.framework_params = framework_params

    def executar_avaliacao(self, X, Y, tamanho_batch_inicial, *args):
        """
        Executa o framework e retorna métricas compatíveis com o Experimento.
        """
        X_init, y_init = X[:tamanho_batch_inicial], Y[:tamanho_batch_inicial]
        X_stream, Y_stream = X[tamanho_batch_inicial:], Y[tamanho_batch_inicial:]

        # Inicialização específica do framework
        tipo_modelo = self.framework_params.get('tipo_modelo_global', RandomForestModelo)
        detector_cls = self.framework_params.get('detector_cls', KSWINDetector)
        detector_params = self.framework_params.get('detector_params', {})

        modelo_inicial, scaler = FrameworkDetector.treinar_modelo_inicial(X_init, y_init, tipo_modelo=tipo_modelo)
        if modelo_inicial is None:
            return float('inf'), 0  # Falha na inicialização

        detector_wrapper = detector_cls(**detector_params)
        janela_dados_recentes = list(zip(X_init[-self.framework_params.get('tamanho_janela', 100):],
                                      y_init[-self.framework_params.get('tamanho_janela', 100):]))

        processor = StreamProcessor(
            modelo_inicial=modelo_inicial,
            detector_wrapper=detector_wrapper,
            scaler=scaler,
            janela_dados_recentes=janela_dados_recentes,
            **{k: v for k, v in self.framework_params.items() if k != 'detector_cls' and k != 'detector_params'}
        )

        # Executa o stream
        start_time = time.time()
        results = processor.processar_stream(
            X_stream=X_stream,
            Y_stream=Y_stream,
            initial_size=tamanho_batch_inicial,
            detector_escolhido=detector_cls.__name__
        )
        execution_time = time.time() - start_time

        # Verifica se temos resultados
        if not results['predicoes_stream']:
            return float('inf'), 0

        # Calcula métricas finais
        mae_final = mean_absolute_error(Y_stream[:len(results['predicoes_stream'])], results['predicoes_stream'])
        qtd_deteccoes = len(results.get('pontos_drift_detectados', []))

        # Para compatibilidade com o experimento, retornamos apenas as métricas chave
        return mae_final, qtd_deteccoes