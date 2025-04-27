import copy
from classes.frameworkClasses.state_processor import StateProcessor
from classes.frameworkDetector.framework_detector import FrameworkDetector


class NormalStateProcessor(StateProcessor):
    """Processador para estado normal"""

    def process(self, processor, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido):

        processor.contador_adicao_pool += 1
        if processor.contador_adicao_pool >= processor.intervalo_adicao_pool:
            novo_modelo_pool = copy.deepcopy(processor.modelo_atual)
            if hasattr(novo_modelo_pool, 'nome'):
                novo_modelo_pool.nome = f"{novo_modelo_pool.nome}_Copy{processor.contador_adicao_pool}"

            pool_atual = processor.pools_por_regime.get(processor.regime_atual, [])
            if not pool_atual:
                processor.pools_por_regime[processor.regime_atual] = [novo_modelo_pool]
            else:
                processor.pools_por_regime[processor.regime_atual] = FrameworkDetector.adicionar_ao_pool(
                    pool_atual, novo_modelo_pool, processor.max_pool_size,
                    processor.janela_dados_recentes, processor.scaler, processor.min_diversidade_erro
                )

            processor.contador_adicao_pool = 0