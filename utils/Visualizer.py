import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Visualizer:
    """
    Classe para visualização de resultados de detecção de drift.
    """

    @staticmethod
    def plotar_resultados(Y, lista_predicoes, labels_algoritmos, deteccoes_por_modelo, tamanho_batch, titulo_plot):
        """
        Plota os resultados de predição e detecção de drift.

        Args:
            Y: Dados reais
            lista_predicoes: Lista de conjuntos de predições
            labels_algoritmos: Rótulos para cada conjunto de predições
            deteccoes_por_modelo: Lista de índices de detecção para cada modelo
            tamanho_batch: Tamanho do batch usado no treinamento
        """
        plt.figure(figsize=(15, 8))
        indices = range(tamanho_batch, tamanho_batch + len(Y[tamanho_batch:]))

        # Plotar valores verdadeiros
        plt.plot(indices, Y[tamanho_batch:tamanho_batch + len(indices)],
                 label="Verdadeiro", linewidth=1.2, color='black')

        # Plotar cada conjunto de previsões
        for i, predicoes in enumerate(lista_predicoes):
            Y_plot = Y[tamanho_batch:tamanho_batch + len(predicoes)]
            predicoes = predicoes[:len(Y_plot)]  # Garantir mesmo tamanho
            label = labels_algoritmos[i] if i < len(labels_algoritmos) else f"Previsões {i+1}"
            plt.plot(indices[:len(predicoes)], predicoes, label=label, linewidth=1.2)

            # Obter detecções para este modelo (se disponíveis)
            modelo_deteccoes = deteccoes_por_modelo[i] if i < len(deteccoes_por_modelo) else []

            # Aumentar o tamanho dos pontos de detecção para este modelo
            if modelo_deteccoes:
                # Verificar se cada ponto de detecção está nos índices válidos
                valid_deteccoes = [d for d in modelo_deteccoes if d < len(Y)]

                if valid_deteccoes:
                    # Usar uma cor diferente para cada modelo
                    cor = plt.cm.tab10(i / 10) if i < 10 else plt.cm.Set3((i-10) / 10)

                    plt.scatter(valid_deteccoes, [Y[d] for d in valid_deteccoes],
                               color=cor, marker='o',
                               label=f"Drift - {label}", zorder=3, s=80)

                    # Destacar áreas pós-retreino com fundo colorido
                    for idx, d in enumerate(valid_deteccoes):
                        if d + tamanho_batch < len(indices):
                            next_end = min(d + tamanho_batch, indices[-1])
                            plt.axvspan(d, next_end, alpha=0.1, color=cor, label='_nolegend_')

                    # Adicionar anotações para mostrar diferenças
                    for d in valid_deteccoes[:3]:  # Limitar a 3 anotações por modelo para não sobrecarregar
                        if d + 5 < len(indices):
                            plt.annotate(f"Retreino {label}",
                                        xy=(d, Y[d]),
                                        xytext=(d+10, Y[d]+0.1 * (i+1)),  # Deslocar verticalmente
                                        arrowprops=dict(facecolor=cor, shrink=0.05, width=1.5),
                                        fontsize=9,
                                        color=cor)

                    print(f"\nDrift detectado para {label} nos índices:", valid_deteccoes)
                else:
                    print(f"\nNenhum drift válido detectado para {label}.")
            else:
                print(f"\nNenhum drift detectado para {label}.")

        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"{titulo_plot}", fontsize=14)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        plt.show()

    @staticmethod
    def plotar_resultados_multi(Y, lista_predicoes, labels_algoritmos, deteccoes_dict, tamanho_batch, detector_or_modelo):
        """
        Plota os resultados com múltiplas detecções.

        Args:
            Y: Dados reais
            lista_predicoes: Lista de conjuntos de predições
            labels_algoritmos: Rótulos para cada conjunto de predições
            deteccoes_dict: Dicionário com índices de detecção para cada modelo
            tamanho_batch: Tamanho do batch usado no treinamento
        """
        plt.figure(figsize=(15, 8))
        indices = range(tamanho_batch, tamanho_batch + len(Y[tamanho_batch:]))

        # Plotar valores verdadeiros
        plt.plot(indices, Y[tamanho_batch:tamanho_batch + len(indices)],
                 label="Verdadeiro", linewidth=1.2, color='black')

        # Plotar cada conjunto de previsões
        for i, predicoes in enumerate(lista_predicoes):
            if i >= len(labels_algoritmos):
                continue

            modelo_nome = labels_algoritmos[i]
            Y_plot = Y[tamanho_batch:tamanho_batch + len(predicoes)]
            predicoes = predicoes[:len(Y_plot)]  # Garantir mesmo tamanho

            # Cor para este modelo
            cor = plt.cm.tab10(i / 10) if i < 10 else plt.cm.Set3((i-10) / 10)

            plt.plot(indices[:len(predicoes)], predicoes, label=modelo_nome, linewidth=1.2, color=cor)

            # Obter detecções para este modelo
            if modelo_nome in deteccoes_dict:
                modelo_deteccoes = deteccoes_dict[modelo_nome]

                if modelo_deteccoes:
                    # Filtrar detecções válidas
                    valid_deteccoes = [d for d in modelo_deteccoes if d < len(Y)]

                    if valid_deteccoes:
                        plt.scatter(valid_deteccoes, [Y[d] for d in valid_deteccoes],
                                  color=cor, marker='o', s=60,
                                  label=f"Drift - {modelo_nome}", zorder=3)

                        # Destacar áreas pós-retreino
                        for d in valid_deteccoes:
                            if d + tamanho_batch < len(indices):
                                next_end = min(d + tamanho_batch, indices[-1])
                                plt.axvspan(d, next_end, alpha=0.1, color=cor, label='_nolegend_')

                        print(f"\nDrift detectado para {modelo_nome} nos índices:", valid_deteccoes)
                    else:
                        print(f"\nNenhum drift válido detectado para {modelo_nome}.")
                else:
                    print(f"\nNenhum drift detectado para {modelo_nome}.")

        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"Predições e Detecção de Drift com Retreino variando os algoritmos de Detecção fixando {detector_or_modelo}", fontsize=14)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        plt.show()

    def visualizar_resultados(resultados, serie_temporal=None, titulo="Resultados da Detecção de Drift"):
        """
        Visualiza os resultados do framework de detecção de drift

        Parâmetros:
        -----------
        resultados : dict
            Dicionário com os resultados do framework
        serie_temporal : pandas.DataFrame, opcional
            Série temporal original, se disponível
        titulo : str
            Título para o gráfico
        """
        # Configurar o estilo do matplotlib para melhor visualização
        plt.style.use('ggplot')

        # Criar figura com subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(titulo, fontsize=16)

        # 1. Gráfico de previsões vs valores reais
        axs[0].plot(resultados["valores_reais"], 'b-', label="Valores Reais", alpha=0.7)
        axs[0].plot(resultados["previsoes"], 'r-', label="Previsões", alpha=0.7)

        # Adicionar pontos de drift como linhas verticais
        for drift in resultados["pontos_drift"]:
            drift_idx = drift - resultados["pontos_drift"][0] if resultados["pontos_drift"] else 0
            axs[0].axvline(x=drift_idx, color='g', linestyle='--', alpha=0.8,
                        label="Drift" if drift == resultados["pontos_drift"][0] else "")

        axs[0].set_title(f"Valores Reais vs Previsões (Detector: {resultados['detector']})", fontsize=12)
        axs[0].set_xlabel("Amostras")
        axs[0].set_ylabel("Valor")
        axs[0].legend(loc='upper right')

        # 2. Gráfico de erros e estados do detector
        axs[1].plot(resultados["erros_predicao"], 'r-', label="Erro de Previsão", alpha=0.7)

        # Adicionar uma linha para o estado do detector
        estados_num = [0 if s == "NORMAL" else 1 if s == "ALERTA" else 2 for s in resultados["estados_detector"]]
        axs[1].plot(estados_num, 'g-', label="Estado do Detector", alpha=0.5)

        # Colocar legenda para os estados
        axs[1].set_yticks([0, 1, 2])
        axs[1].set_yticklabels(["NORMAL", "ALERTA", "MUDANÇA"])

        axs[1].set_title("Erros de Previsão e Estados do Detector", fontsize=12)
        axs[1].set_xlabel("Amostras")
        axs[1].set_ylabel("Erro / Estado")
        axs[1].legend(loc='upper right')

        # 3. Métricas de desempenho ao longo do tempo
        if resultados["metricas_rmse"]:
            pontos_rmse, valores_rmse = zip(*resultados["metricas_rmse"])
            pontos_rmse = [p - resultados["metricas_rmse"][0][0] for p in pontos_rmse]
            axs[2].plot(pontos_rmse, valores_rmse, 'b-', label="RMSE", alpha=0.7)

        if resultados["metricas_mae"]:
            pontos_mae, valores_mae = zip(*resultados["metricas_mae"])
            pontos_mae = [p - resultados["metricas_mae"][0][0] for p in pontos_mae]
            axs[2].plot(pontos_mae, valores_mae, 'r-', label="MAE", alpha=0.7)

        if resultados["metricas_r2"]:
            pontos_r2, valores_r2 = zip(*resultados["metricas_r2"])
            pontos_r2 = [p - resultados["metricas_r2"][0][0] for p in pontos_r2]
            axs[2].plot(pontos_r2, valores_r2, 'g-', label="R²", alpha=0.7)

        axs[2].set_title("Métricas de Desempenho", fontsize=12)
        axs[2].set_xlabel("Amostras")
        axs[2].set_ylabel("Valor")
        axs[2].legend(loc='upper right')

        # Informações resumidas
        n_drifts = len(resultados["pontos_drift"])
        rmse_final = np.sqrt(mean_squared_error(resultados["valores_reais"], resultados["previsoes"]))
        mae_final = mean_absolute_error(resultados["valores_reais"], resultados["previsoes"])

        stats_text = (f"Detector: {resultados['detector']}\n"
                    f"Drifts detectados: {n_drifts}\n"
                    f"RMSE Final: {rmse_final:.4f}\n"
                    f"MAE Final: {mae_final:.4f}\n"
                    f"Tamanho pool: {len(resultados['pool_modelos'])}")

        # Adicionar texto de estatísticas ao canto superior direito do gráfico
        plt.figtext(0.91, 0.85, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.subplots_adjust(top=0.9)
        plt.show()

        # Estatísticas adicionais
        print(f"\nDetector: {resultados['detector']}")
        print(f"Total de amostras processadas: {len(resultados['valores_reais'])}")
        print(f"Número de drifts detectados: {n_drifts}")
        if n_drifts > 0:
            print(f"Pontos de drift: {resultados['pontos_drift']}")
        print(f"RMSE final: {rmse_final:.4f}")
        print(f"MAE final: {mae_final:.4f}")

        # Se houver pelo menos 2 drifts detectados, calcular intervalo médio entre drifts
        if n_drifts >= 2:
            intervalos = [resultados["pontos_drift"][i+1] - resultados["pontos_drift"][i]
                        for i in range(n_drifts-1)]
            print(f"Intervalo médio entre drifts: {np.mean(intervalos):.1f} amostras")
