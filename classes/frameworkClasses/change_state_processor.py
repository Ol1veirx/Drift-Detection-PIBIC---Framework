import copy

from classes.frameworkClasses.state_processor import StateProcessor


class ChangeStateProcessor(StateProcessor):
    """Processador para estado de mudança (drift)"""

    def process(self, processor, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido):
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        processor.pontos_drift_detectados.append(indice_global)
        print(f"\n!!! Drift detectado no índice {indice_global} (Detector: {detector_escolhido}, Regime: {processor.regime_atual}) !!!")

        pool_atual = processor.pools_por_regime.get(processor.regime_atual, [])
        if not pool_atual:
            pool_atual = processor.pools_por_regime.get(0, [])
            print(f"  ⚠️ Usando pool do regime 0 como fallback")

        print(f"  Avaliando {len(pool_atual)} modelos do pool...")
        melhor_do_pool = FrameworkDetector.selecionar_melhor_modelo(pool_atual, processor.janela_dados_recentes, processor.scaler)

        if melhor_do_pool:
            erro_atual = FrameworkDetector.desempenho(processor.modelo_atual, processor.janela_dados_recentes, processor.scaler)
            erro_melhor = FrameworkDetector.desempenho(melhor_do_pool, processor.janela_dados_recentes, processor.scaler)
            print(f"  Desempenho: atual={erro_atual:.4f}, melhor do pool={erro_melhor:.4f}")

            if erro_melhor < erro_atual:
                processor.modelo_atual = melhor_do_pool
                print(f"  ✓ Novo modelo ativo: '{processor.modelo_atual.nome if hasattr(processor.modelo_atual, 'nome') else type(processor.modelo_atual).__name__}'")

            processor.modelo_a_manter_do_pool_anterior = copy.deepcopy(processor.modelo_atual)
        else:
            processor.modelo_a_manter_do_pool_anterior = copy.deepcopy(processor.modelo_atual)
            print("  ⚠️ Mantendo modelo atual (pool sem alternativas)")

        processor.pools_por_regime[processor.regime_atual] = FrameworkDetector.gerenciar_pool(
            pool_atual, processor.janela_dados_recentes, processor.scaler,
            estado_detector="MUDANÇA", modelo_atual=processor.modelo_atual, max_pool_size=processor.max_pool_size
        )

        processor.drift_detectado_flag = True
        processor.buffer_novo_conceito = [(x_t, y_t)]
        processor.modelo_transicao = processor._criar_modelo_transicao()

        print(f"  Iniciando coleta para retreino ({processor.observacoes_novo_conceito} amostras)")
