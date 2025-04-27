import numpy as np

from classes.frameworkClasses.state_processor import StateProcessor


class AlertStateProcessor(StateProcessor):
    """Processador para estado de alerta"""

    def process(self, processor, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido):
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        pool_atual = processor.pools_por_regime.get(processor.regime_atual, [])

        if len(pool_atual) > 1:
            melhor_do_pool = FrameworkDetector.selecionar_melhor_modelo(pool_atual, processor.janela_dados_recentes, processor.scaler)

            if melhor_do_pool and melhor_do_pool is not processor.modelo_atual:
                erro_atual = FrameworkDetector.desempenho(processor.modelo_atual, processor.janela_dados_recentes, processor.scaler)
                erro_melhor = FrameworkDetector.desempenho(melhor_do_pool, processor.janela_dados_recentes, processor.scaler)

                if erro_melhor < erro_atual * processor.threshold_melhoria_alerta:
                    processor.modelo_atual = melhor_do_pool
                    print(f"\n  🔄 Troca durante alerta: '{processor.modelo_atual.nome if hasattr(processor.modelo_atual, 'nome') else type(processor.modelo_atual).__name__}' (erro: {erro_melhor:.4f})")

        if hasattr(processor.modelo_atual, "partial_fit"):
            processor.modelo_atual.partial_fit(x_t_scaled, np.array([y_t]))

        if i % 10 == 0 and hasattr(processor.modelo_atual, "criar_variante"):
            variante = processor.modelo_atual.criar_variante(mutation_strength=0.1)
            if variante:
                processor.pools_por_regime[processor.regime_atual] = FrameworkDetector.adicionar_ao_pool(
                    processor.pools_por_regime.get(processor.regime_atual, []), variante, processor.max_pool_size,
                    processor.janela_dados_recentes, processor.scaler, processor.min_diversidade_erro
                )