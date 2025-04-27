import numpy as np
from classes.frameworkClasses.state_processor import StateProcessor


class DriftCollectionProcessor(StateProcessor):
    """Processador para coleta de dados após drift"""

    def process(self, processor, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido):
        from classes.frameworkDetector.framework_detector import FrameworkDetector

        processor.contador_novo_conceito += 1
        processor.buffer_novo_conceito.append((x_t, y_t))

        if hasattr(processor.modelo_atual, "partial_fit"):
            processor.modelo_atual.partial_fit(x_t_scaled, np.array([y_t]))

        if processor.contador_novo_conceito >= processor.observacoes_novo_conceito:
            print(f"\n--- Retreinando Modelo com {processor.observacoes_novo_conceito} obs. do novo conceito ---")
            X_novo_list = np.array([x for x, y in processor.buffer_novo_conceito])
            y_novo_array = np.array([y for x, y in processor.buffer_novo_conceito])

            novo_modelo = FrameworkDetector.treinar_novo_conceito(
                X_novo_list, y_novo_array, processor.scaler, tipo_modelo=processor.tipo_modelo_global
            )

            if novo_modelo:
                if hasattr(novo_modelo, 'nome'):
                    novo_modelo.nome = f"{processor.tipo_modelo_global.__name__}_R{processor.regime_atual}_D{len(processor.pontos_drift_detectados)}"

                processor.modelo_atual = novo_modelo
                processor.pools_por_regime[processor.regime_atual] = [processor.modelo_atual]

                if processor.modelo_a_manter_do_pool_anterior and processor.modelo_a_manter_do_pool_anterior is not processor.modelo_atual:
                    erro_atual = FrameworkDetector.desempenho(processor.modelo_atual, processor.janela_dados_recentes, processor.scaler)
                    erro_anterior = FrameworkDetector.desempenho(processor.modelo_a_manter_do_pool_anterior, processor.janela_dados_recentes, processor.scaler)

                    if abs(erro_atual - erro_anterior) >= processor.min_diversidade_erro:
                        processor.pools_por_regime[processor.regime_atual].append(processor.modelo_a_manter_do_pool_anterior)
                        print(f"  ✓ Modelo anterior mantido no pool (diverso)")
                    else:
                        print(f"  ⚠️ Modelo anterior descartado (não diverso)")

                print(f"  ✓ Novo modelo ativo: '{processor.modelo_atual.nome if hasattr(processor.modelo_atual, 'nome') else type(processor.modelo_atual).__name__}'")
            else:
                print(f"  ⚠️ Falha no retreino. Mantendo modelo atual.")

            processor.drift_detectado_flag = False
            processor.contador_novo_conceito = 0
            processor.buffer_novo_conceito = []
            processor.contador_adicao_pool = 0
            processor.modelo_transicao = None
            print("--- Retomando operação normal ---")