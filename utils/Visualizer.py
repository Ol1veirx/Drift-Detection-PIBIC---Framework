import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    def analisar_e_visualizar_resultados_stream(
        initial_size,
        erros_predicao_stream,
        estados_detector_stream,
        pontos_drift_detectados,
        metricas_rmse_stream,
        metricas_mae_stream,
        metricas_r2_stream,
        modelo_ativo_ao_longo_do_tempo,
        detector_escolhido,
        serie_escolhida,
        tamanho_pool_ao_longo_do_tempo
    ):
        """
        Analisa e visualiza os resultados do processamento do stream,
        incluindo erros, estados do detector, métricas de desempenho e uso de modelos.

        Args:
            initial_size (int): Tamanho do conjunto de treinamento inicial.
            erros_predicao_stream (list): Lista de erros de predição absolutos.
            estados_detector_stream (list): Lista de estados do detector ('NORMAL', 'ALERTA', 'MUDANÇA').
            pontos_drift_detectados (list): Lista de índices onde drifts foram detectados.
            metricas_rmse_stream (list): Lista de tuplas (índice, RMSE) calculadas periodicamente.
            metricas_mae_stream (list): Lista de tuplas (índice, MAE) calculadas periodicamente.
            metricas_r2_stream (list): Lista de tuplas (índice, R²) calculadas periodicamente.
            modelo_ativo_ao_longo_do_tempo (list): Lista com os nomes dos modelos ativos em cada passo.
            detector_escolhido (str): Nome do detector utilizado.
            serie_escolhida (str): Nome da série temporal processada.
            tamanho_pool_ao_longo_do_tempo (list): Lista com o tamanho do pool de modelos ao longo do tempo.
        """
        print("\n=== Análise dos Resultados do Stream ===")

        # Criar eixo de tempo para os resultados do stream
        stream_indices = np.arange(initial_size, initial_size + len(erros_predicao_stream))

        # Verificar se há resultados para plotar
        if not erros_predicao_stream:
            print("Nenhum resultado do stream para analisar.")
        else:
            plt.figure(figsize=(15, 12)) # Aumentar a altura da figura para acomodar 4 gráficos

            # --- Gráfico 1: Erro ---
            plt.subplot(4, 1, 1) # <-- 4 linhas, 1 coluna, 1º gráfico
            plt.plot(stream_indices, erros_predicao_stream, label='Erro de Predição (Absoluto)', color='red', alpha=0.7)
            plt.title(f'Erro, Estado, Métricas e Pool ({detector_escolhido}) na Série {serie_escolhida}')
            plt.ylabel('Erro Absoluto')
            plt.grid(True, alpha=0.5)
            drift_label_added = False
            for ponto in pontos_drift_detectados:
                plt.axvline(x=ponto, color='black', linestyle='--', linewidth=1.5, label='Drift Detectado' if not drift_label_added else "")
                drift_label_added = True
            if drift_label_added: plt.legend()
            plt.gca().axes.xaxis.set_ticklabels([]) # Oculta rótulos do eixo x para gráficos superiores

            # --- Gráfico 2: Estado do Detector ---
            plt.subplot(4, 1, 2) # <-- 4 linhas, 1 coluna, 2º gráfico
            estado_map = {"NORMAL": 0, "ALERTA": 1, "MUDANÇA": 2}
            estados_numericos = [estado_map.get(s, -1) for s in estados_detector_stream]
            plt.plot(stream_indices, estados_numericos, label='Estado do Detector', color='green')
            plt.yticks([0, 1, 2], ['NORMAL', 'ALERTA', 'MUDANÇA'])
            plt.ylabel('Estado')
            plt.grid(True, alpha=0.5)
            for ponto in pontos_drift_detectados:
                plt.axvline(x=ponto, color='black', linestyle='--', linewidth=1.5)
            plt.legend()
            plt.gca().axes.xaxis.set_ticklabels([]) # Oculta rótulos do eixo x

            # --- Gráfico 3: Métricas de Desempenho ---
            plt.subplot(4, 1, 3) # <-- 4 linhas, 1 coluna, 3º gráfico
            ax_primary = plt.gca()
            legend_handles_primary = []
            if metricas_rmse_stream:
                idx_rmse, val_rmse = zip(*metricas_rmse_stream)
                line_rmse, = ax_primary.plot(idx_rmse, val_rmse, label='RMSE (Janela)', color='blue')
                legend_handles_primary.append(line_rmse)
            if metricas_mae_stream:
                idx_mae, val_mae = zip(*metricas_mae_stream)
                line_mae, = ax_primary.plot(idx_mae, val_mae, label='MAE (Janela)', color='purple')
                legend_handles_primary.append(line_mae)
            ax_primary.set_ylabel('RMSE / MAE', color='blue')
            ax_primary.tick_params(axis='y', labelcolor='blue')
            ax_primary.legend(handles=legend_handles_primary, loc='lower left')
            ax_primary.grid(True, alpha=0.5)
            ax_primary.set_title('Métricas de Desempenho (Calculadas em Janela Deslizante)') # Adiciona título específico
            ax_primary.axes.xaxis.set_ticklabels([]) # Oculta rótulos do eixo x

            ax_secondary = None # Inicializa para verificar depois
            if metricas_r2_stream:
                ax_secondary = ax_primary.twinx()
                idx_r2, val_r2 = zip(*metricas_r2_stream)
                line_r2, = ax_secondary.plot(idx_r2, val_r2, label='R² (Janela)', color='orange', linestyle=':')
                ax_secondary.set_ylabel('R²', color='orange')
                ax_secondary.tick_params(axis='y', labelcolor='orange')
                ax_secondary.legend(handles=[line_r2], loc='lower right')

            for ponto in pontos_drift_detectados:
                ax_primary.axvline(x=ponto, color='black', linestyle='--', linewidth=1.5)
                if ax_secondary: ax_secondary.axvline(x=ponto, color='black', linestyle='--', linewidth=1.5)


            # --- Gráfico 4: Tamanho do Pool ---
            plt.subplot(4, 1, 4) # <-- 4 linhas, 1 coluna, 4º gráfico
            if tamanho_pool_ao_longo_do_tempo:
                plt.plot(stream_indices, tamanho_pool_ao_longo_do_tempo, label='Tamanho do Pool', color='cyan')
                plt.ylabel('Número de Modelos')
                plt.xlabel('Amostra (Índice Global)') # Eixo X apenas no último gráfico
                plt.title('Evolução do Tamanho do Pool de Modelos')
                plt.grid(True, alpha=0.5)
                for ponto in pontos_drift_detectados:
                    plt.axvline(x=ponto, color='black', linestyle='--', linewidth=1.5)
                # Ajustar limites do eixo Y para inteiros se apropriado
                max_pool_val = max(tamanho_pool_ao_longo_do_tempo) if tamanho_pool_ao_longo_do_tempo else 1
                plt.ylim(0, max_pool_val + 1)
                plt.yticks(range(0, int(max_pool_val + 1))) # Garante ticks inteiros
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'Dados do tamanho do pool não disponíveis', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


            plt.tight_layout() # Ajusta o espaçamento para evitar sobreposição de títulos/rótulos
            plt.show()

        # --- 3. Análise Adicional (Impressão no Console) ---
        if erros_predicao_stream:
            # Calcula métricas globais sobre todo o stream
            erros_np = np.array(erros_predicao_stream)
            mse_global_stream = np.mean(erros_np**2)
            mae_global_stream = np.mean(erros_np) # Erro já é absoluto no seu loop principal
            print(f"\nMétricas Globais no Stream:")
            print(f"  MSE: {mse_global_stream:.4f}")
            print(f"  MAE: {mae_global_stream:.4f}")

        if modelo_ativo_ao_longo_do_tempo:
            # Conta quantas vezes cada modelo foi usado
            modelos_usados = pd.Series(modelo_ativo_ao_longo_do_tempo).value_counts()
            print("\nContagem de Uso dos Modelos Durante o Stream:")
            print(modelos_usados)

    @staticmethod
    def visualizar_previsoes_vs_real(initial_size, Y_stream, predicoes_stream, pontos_drift_detectados, serie_escolhida):
        """
        Visualiza as previsões do modelo vs. valores reais ao longo da série temporal.

        Args:
            initial_size: Tamanho do conjunto de treinamento inicial
            Y_stream: Valores reais do stream
            predicoes_stream: Previsões feitas pelo modelo
            pontos_drift_detectados: Pontos onde drifts foram detectados
            serie_escolhida: Nome da série temporal
        """
        plt.figure(figsize=(15, 6))

        # Converter índices para escala global
        stream_indices = [initial_size + i for i in range(len(Y_stream))]

        # Garantir que as previsões sejam valores escalares
        predicoes_stream_flat = []
        for pred in predicoes_stream:
            # Se for um array ou lista, pegar o primeiro elemento
            if isinstance(pred, (list, np.ndarray)):
                predicoes_stream_flat.append(float(pred[0]) if len(pred) > 0 else float('nan'))
            else:
                # Caso seja um escalar, converter para float
                predicoes_stream_flat.append(float(pred))

        # Plotar os valores reais
        plt.plot(stream_indices, Y_stream, label='Valor Real', color='blue', alpha=0.7, linewidth=1.5)

        # Plotar as previsões (usando a versão normalizada)
        plt.plot(stream_indices, predicoes_stream_flat, label='Previsão do Modelo', color='red', alpha=0.8, linestyle='--', linewidth=1.5)

        plt.title(f'Valor Real vs. Previsão do Modelo na Série {serie_escolhida}')
        plt.xlabel('Amostra (Índice Global)')
        plt.ylabel('Valor')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Adicionar marcações verticais para os drifts detectados
        for drift_point in pontos_drift_detectados:
            plt.axvline(x=drift_point, color='black', linestyle=':', alpha=0.7, linewidth=1.2)

        # Adicionar texto indicando os drifts
        if pontos_drift_detectados:
            plt.text(0.02, 0.02, 'Drift Detectado', transform=plt.gca().transAxes,
                    color='black', alpha=0.7, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()