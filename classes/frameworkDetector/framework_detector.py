from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from classes.modelosOffline.KneighborsRegressorModelo import KNeighborsRegressorModelo
from classes.modelosOffline.LassoRegressionModelo import LassoRegressionModelo
from classes.modelosOffline.LinearRegressionModelo import LinearRegressionModelo
from classes.modelosOffline.RandomForestModelo import RandomForestModelo
from classes.modelosOffline.RidgeRegressionModelo import RidgeRegressionModelo
from classes.modelosOffline.SVRModelo import SVRModelo
from utils.SeriesProcessor import SeriesProcessor


class FrameworkDetector:
    @staticmethod
    def treinar_modelos_iniciais(X, y):
        """Treina e retorna uma lista de modelos iniciais"""
        modelos = []

        # Lista de modelos para tentar treinar
        modelos_para_treinar = [
            #(LinearRegressionModelo(), "Regressão Linear"),
            #(RidgeRegressionModelo(), "Ridge"),
            #(LassoRegressionModelo(), "Lasso"),
            (SVRModelo(), "SVR"),
            (RandomForestModelo(), "Random Forest"),
            (KNeighborsRegressorModelo(), "KNN")
        ]

        print("Iniciando treinamento dos modelos iniciais...")

        # Tenta treinar cada modelo com tratamento de erro
        for modelo, nome in modelos_para_treinar:
            try:
                print(f"Treinando modelo {nome}...")
                modelo_treinado = modelo.treinar(X, y)

                if modelo_treinado is not None:
                    print(f"✅ Modelo {nome} treinado com sucesso")
                    modelos.append(modelo_treinado)
                else:
                    print(f"❌ Falha ao treinar modelo {nome} (retornou None)")
            except Exception as e:
                print(f"❌ Erro ao treinar modelo {nome}: {e}")

        # Verificação final para garantir que pelo menos um modelo foi criado
        if len(modelos) == 0:
            print("⚠️ AVISO: Nenhum modelo foi treinado com sucesso. Tentando modelo mais simples...")
            try:
                modelo_simples = LinearRegressionModelo()
                modelo_treinado = modelo_simples.treinar(X, y)
                if modelo_treinado is not None:
                    modelos.append(modelo_treinado)
                    print("✅ Modelo de fallback treinado com sucesso")
            except Exception as e:
                print(f"❌ Erro crítico: Não foi possível treinar nem mesmo o modelo de fallback: {e}")

        print(f"✓ Total de modelos treinados com sucesso: {len(modelos)}")
        for i, modelo in enumerate(modelos):
            nome = modelo.__class__.__name__ if hasattr(modelo, '__class__') else "Desconhecido"
            print(f"  {i+1}. {nome}")

        return modelos

    @staticmethod
    def selecionar_melhor_modelo(pool, janela):
        """Seleciona o melhor modelo do pool com base no erro quadrático médio"""
        # Filtra modelos None do pool
        pool_filtrado = [modelo for modelo in pool if modelo is not None]

        if not pool_filtrado:
            print("ERRO: Não há modelos válidos no pool!")
            return None

        X_janela = [x for x, y in janela]
        y_janela = [y for x, y in janela]

        # Lista para armazenar erros com índices correspondentes
        erros_com_indices = []

        for i, modelo in enumerate(pool_filtrado):
            try:
                erro = mean_squared_error(y_janela, modelo.prever(X_janela))
                erros_com_indices.append((erro, i))
            except Exception as e:
                print(f"Erro ao avaliar modelo #{i}: {e}")

        if not erros_com_indices:
            print("ERRO: Nenhum modelo pôde ser avaliado!")
            return pool_filtrado[0] if pool_filtrado else None

        menor_erro, melhor_indice = min(erros_com_indices)
        return pool_filtrado[melhor_indice]

    @staticmethod
    def desempenho(modelo, janela):
        """Avalia o desempenho do modelo na janela atual"""
        X_janela = [x for x, y in janela]
        y_janela = [y for x, y in janela]
        return mean_squared_error(y_janela, modelo.prever(X_janela))

    @staticmethod
    def adicionar_a_janela(janela, dado, tamanho_max=100):
        """Adiciona um novo dado à janela, removendo o mais antigo se necessário"""
        janela.append(dado)
        if len(janela) > tamanho_max:
            janela.pop(0)
        return janela

    def get_state(detector):
        """
        Função auxiliar para mapear o estado do detector para NORMAL, ALERTA ou MUDANÇA.
        Funciona com diferentes tipos de detectores.
        """
        # DDM
        if hasattr(detector, 'in_concept_change'):
            if detector.in_concept_change:
                return "MUDANÇA"
            elif detector.in_warning_zone:
                return "ALERTA"
            else:
                return "NORMAL"

        # ADWIN
        elif hasattr(detector, 'detected_change'):
            if detector.detected_change():
                return "MUDANÇA"
            # ADWIN geralmente não tem estado de alerta, então verificamos a magnitude da diferença
            elif hasattr(detector, 'estimation'):
                # Esta é uma implementação aproximada - você precisará ajustar baseado na sua implementação
                if abs(detector.estimation - detector._last_estimation) > detector.delta / 2:
                    return "ALERTA"
            return "NORMAL"

        # ECDD ou outros detectores
        elif hasattr(detector, '_current_state') or hasattr(detector, 'current_state'):
            state = getattr(detector, '_current_state', None) or getattr(detector, 'current_state', None)
            if state in ["NORMAL", "ALERTA", "MUDANÇA"]:
                return state
            # Mapear outros estados se necessário

        # Caso genérico - tenta várias abordagens comuns
        else:
            # Tentar acessar variáveis/métodos comuns que indicam mudança
            # Estas são apenas sugestões - ajuste conforme suas implementações específicas

            # Verificar se há métodos específicos
            if hasattr(detector, 'detected_drift') and callable(getattr(detector, 'detected_drift')):
                if detector.detected_drift():
                    return "MUDANÇA"

            if hasattr(detector, 'warning_detected') and callable(getattr(detector, 'warning_detected')):
                if detector.warning_detected():
                    return "ALERTA"

            # Verificar atributos comuns
            if hasattr(detector, 'drift_detected') and detector.drift_detected:
                return "MUDANÇA"

            if hasattr(detector, 'warning') and detector.warning:
                return "ALERTA"

        # Se nenhuma das verificações anteriores funcionar, assume estado normal
        return "NORMAL"

    def calcular_erro_relativo(y_true, y_pred):
        """Calcula o erro relativo médio absoluto"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))

    # 1. Dados Sintéticos com Drift Controlado
    def gerar_dados_sinteticos(n_amostras=5000, n_features=5, n_drifts=3, seed=None):
        """Gera dados sintéticos com drift controlado para regressão"""
        if seed is not None:
            np.random.seed(seed)

        X = np.random.randn(n_amostras, n_features)
        y = np.zeros(n_amostras)

        # Pontos de drift
        drift_points = [int(i * n_amostras / (n_drifts + 1)) for i in range(1, n_drifts + 1)]

        # Regras para cada segmento
        def regra_1(x):
            return 5 + 2 * x[:, 0] + 0.5 * x[:, 1] + np.random.randn(len(x)) * 0.5

        def regra_2(x):
            return 10 - 1.5 * x[:, 0] + 1 * x[:, 1] + np.random.randn(len(x)) * 0.8

        def regra_3(x):
            return 2 + 3 * x[:, 0] + 2 * x[:, 2] + np.random.randn(len(x)) * 0.3

        def regra_4(x):
            return 8 + 0.1 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * x[:, 3] + np.random.randn(len(x)) * 0.6

        regras = [regra_1, regra_2, regra_3, regra_4]

        # Aplicar regras aos segmentos
        regra_atual = 0
        inicio = 0

        for ponto in drift_points:
            # Aplicar regra atual até o ponto de drift
            indices = np.arange(inicio, ponto)
            y[indices] = regras[regra_atual](X[indices])

            # Mudar para próxima regra
            regra_atual = (regra_atual + 1) % len(regras)
            inicio = ponto

        # Aplicar regra final no último segmento
        indices = np.arange(inicio, n_amostras)
        y[indices] = regras[regra_atual](X[indices])

        return X, y, drift_points

    def analisar_pool_modelos(resultados):
        """
        Analisa o pool de modelos e exibe informações sobre cada um
        """
        print(f"\n=== Pool de Modelos ({len(resultados['pool_modelos'])} modelos) ===")

        for i, modelo in enumerate(resultados['pool_modelos']):
            print(f"\nModelo #{i+1}:")

            # Mostrar tipo do modelo
            if hasattr(modelo, 'nome'):
                print(f"Tipo: {modelo.nome}")
            else:
                print(f"Tipo: {type(modelo).__name__}")

            # Mostrar parâmetros, se disponíveis
            if hasattr(modelo, 'modelo') and hasattr(modelo.modelo, 'get_params'):
                params = modelo.modelo.get_params()
                print("Parâmetros:")
                for param, valor in params.items():
                    print(f"  {param}: {valor}")

            # Mostrar coeficientes para modelos lineares
            if hasattr(modelo, 'modelo'):
                if hasattr(modelo.modelo, 'coef_'):
                    coefs = modelo.modelo.coef_
                    print(f"Coeficientes: {coefs[:5]}{'...' if len(coefs) > 5 else ''}")
                if hasattr(modelo.modelo, 'intercept_'):
                    print(f"Intercepto: {modelo.modelo.intercept_}")

        # Modelo final usado
        print("\n=== Modelo Final ===")
        modelo_final = resultados['modelo_final']

        if hasattr(modelo_final, 'nome'):
            print(f"Tipo: {modelo_final.nome}")
        else:
            print(f"Tipo: {type(modelo_final).__name__}")

        # Avaliar desempenho do modelo final nos últimos dados
        valores_reais = resultados['valores_reais'][-50:]  # Últimas 50 amostras
        previsoes = resultados['previsoes'][-50:]

        rmse = np.sqrt(mean_squared_error(valores_reais, previsoes))
        r2 = r2_score(valores_reais, previsoes)

        print(f"RMSE nas últimas 50 amostras: {rmse:.4f}")
        print(f"R² nas últimas 50 amostras: {r2:.4f}")

    def comparar_modelos_pool(resultados, X_teste, y_teste, num_modelos=3):
        """
        Compara visualmente os modelos do pool (versão leve)
        """
        # Reduzir o tamanho da figura e usar menos amostras
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=80)

        # Selecionar apenas alguns modelos representativos
        modelos_para_comparar = resultados['pool_modelos']

        # Limitar o número de modelos para comparação
        if len(modelos_para_comparar) > num_modelos:
            # Pegar primeiro, último e intermediário
            indices = [0]
            if num_modelos >= 2:
                indices.append(len(modelos_para_comparar)-1)
            if num_modelos >= 3:
                indices.append(len(modelos_para_comparar)//2)
            modelos_para_comparar = [modelos_para_comparar[i] for i in indices]

        # Métricas para cada modelo
        rmse_valores = []
        nomes_modelos = []

        # Reduzir dados de teste para visualização
        n_amostras = min(50, len(X_teste))
        X_amostra = X_teste[:n_amostras]
        y_amostra = y_teste[:n_amostras]

        # Calcular previsões de cada modelo
        for i, modelo in enumerate(modelos_para_comparar):
            try:
                # Previsões do modelo
                y_pred = modelo.prever(X_amostra)
                rmse = np.sqrt(mean_squared_error(y_amostra, y_pred))
                rmse_valores.append(rmse)

                # Nome simplificado do modelo
                if hasattr(modelo, 'nome'):
                    nome = modelo.nome.split()[0]  # Só primeira palavra
                else:
                    nome = f"M{i+1}"
                nomes_modelos.append(nome)

                # Plotar só uma linha, sem marcadores
                axs[0].plot(y_pred, label=f"{nome} ({rmse:.2f})", linewidth=1)
            except Exception as e:
                print(f"Erro com modelo {i}: {e}")

        # Adicionar valores reais como referência
        axs[0].plot(y_amostra, 'k-', label='Real', linewidth=2)
        axs[0].set_title('Previsões (50 amostras)')
        axs[0].set_xlabel('Amostra')
        axs[0].set_ylabel('Valor')
        axs[0].legend(loc='best', fontsize='small')

        # Gráfico de barras simplificado
        if rmse_valores:
            axs[1].bar(nomes_modelos, rmse_valores)
            axs[1].set_title('RMSE por Modelo')
            axs[1].set_xlabel('Modelo')
            axs[1].set_ylabel('RMSE')

        plt.tight_layout()
        # Salvar em vez de mostrar (opcional)
        # plt.savefig('comparacao_modelos.png', dpi=100)
        plt.show()

        # Resumo textual em vez de gráfico extenso
        print("\n=== Resumo dos Modelos ===")
        for i, (nome, rmse) in enumerate(zip(nomes_modelos, rmse_valores)):
            print(f"{nome}: RMSE={rmse:.4f}")

    def analisar_overfitting(resultados, X, y, tamanho_inicial):
        """
        Análise simplificada de overfitting
        """
        # Usar menos dados para a análise
        X_treino = X[:tamanho_inicial]
        y_treino = y[:tamanho_inicial]

        # Usar apenas uma amostra do teste para acelerar
        amostra_tamanho = min(200, len(X) - tamanho_inicial)
        X_teste = X[tamanho_inicial:tamanho_inicial+amostra_tamanho]
        y_teste = y[tamanho_inicial:tamanho_inicial+amostra_tamanho]

        modelo_final = resultados['modelo_final']

        # Avalia o modelo nos dados de treinamento e teste
        try:
            y_treino_pred = modelo_final.prever(X_treino)
            rmse_treino = np.sqrt(mean_squared_error(y_treino, y_treino_pred))
            r2_treino = r2_score(y_treino, y_treino_pred)

            # Usar apenas a amostra dos dados de teste
            y_teste_pred = modelo_final.prever(X_teste)
            rmse_teste = np.sqrt(mean_squared_error(y_teste, y_teste_pred))
            r2_teste = r2_score(y_teste, y_teste_pred)

            # Calcula razão (indicador de overfitting)
            razao_rmse = rmse_teste / rmse_treino if rmse_treino > 0 else float('inf')

            print("\n=== Análise de Overfitting ===")
            print(f"RMSE treino: {rmse_treino:.4f}, teste: {rmse_teste:.4f}, razão: {razao_rmse:.2f}")
            print(f"R² treino: {r2_treino:.4f}, teste: {r2_teste:.4f}")

            # Interpretação simplificada
            if razao_rmse > 1.5:
                print("⚠️ Possível overfitting detectado!")
            elif razao_rmse < 1.1:
                print("✅ Sem overfitting significativo.")
            else:
                print("ℹ️ Diferença aceitável entre treino e teste.")

            # Gráfico único e simples
            plt.figure(figsize=(8, 4), dpi=80)

            # Usar menos pontos para o scatter plot
            n_pontos = min(50, len(y_treino))
            indices_treino = np.random.choice(len(y_treino), n_pontos, replace=False)

            plt.scatter(y_treino[indices_treino], y_treino_pred[indices_treino],
                    alpha=0.6, label='Treino', s=20)

            n_pontos_teste = min(50, len(y_teste))
            indices_teste = np.random.choice(len(y_teste), n_pontos_teste, replace=False)

            plt.scatter(y_teste[indices_teste], y_teste_pred[indices_teste],
                    alpha=0.6, label='Teste', s=20)

            # Linha ideal
            min_val = min(y_treino.min(), y_teste.min())
            max_val = max(y_treino.max(), y_teste.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

            plt.xlabel('Valores Reais')
            plt.ylabel('Previsões')
            plt.title('Previsões vs Valores Reais')
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Erro na análise de overfitting: {e}")