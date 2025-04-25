import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import traceback
import copy # Importar copy para deepcopy

# Importar seus wrappers de modelo
from classes.modelosOffline.KneighborsRegressorModelo import KNeighborsRegressorModelo
from classes.modelosOffline.LassoRegressionModelo import LassoRegressionModelo
from classes.modelosOffline.LinearRegressionModelo import LinearRegressionModelo
from classes.modelosOffline.RandomForestModelo import RandomForestModelo
from classes.modelosOffline.RidgeRegressionModelo import RidgeRegressionModelo
from classes.modelosOffline.SVRModelo import SVRModelo
# Seus wrappers de detector já são importados no notebook/script principal

class FrameworkDetector:
    """
    Classe utilitária estática contendo a lógica principal para o framework
    de detecção de drift e adaptação com pool de modelos.
    """

    @staticmethod
    def treinar_modelos_iniciais(X, y):
        """
        Treina uma lista inicial diversificada de modelos e aplica escalonamento.

        Args:
            X (np.ndarray): Features de treinamento inicial.
            y (np.ndarray): Target de treinamento inicial.

        Returns:
            tuple: Uma tupla contendo:
                - list: A lista de modelos wrapper treinados.
                - StandardScaler: O scaler ajustado nos dados X.
        """
        modelos_treinados = []
        scaler = StandardScaler()

        print("Ajustando Scaler nos dados iniciais...")
        try:
            # Ajustar e transformar os dados iniciais
            X_scaled = scaler.fit_transform(X)
            print("✓ Scaler ajustado e aplicado.")
        except Exception as e:
            print(f"❌ Erro ao ajustar/aplicar scaler: {e}. Continuando sem scaling.")
            X_scaled = X # Usar dados originais se scaling falhar

        # Lista de wrappers de modelos para tentar treinar
        # Adicione ou remova conforme necessário
        modelos_para_treinar = [
            #(LinearRegressionModelo(), "Regressão Linear"), # Menos sensível ao scaling
            #(RidgeRegressionModelo(), "Ridge"),           # Menos sensível ao scaling
            (SVRModelo(), "SVR"),                         # PRECISA de scaling
            (RandomForestModelo(), "Random Forest"),     # Menos sensível, mas pode ajudar
            (KNeighborsRegressorModelo(), "KNN")         # PRECISA de scaling
        ]

        print("\nIniciando treinamento dos modelos iniciais...")
        for modelo_wrapper_instancia, nome in modelos_para_treinar:
            try:
                print(f"  Treinando modelo {nome}...")
                # Chama o método 'treinar' da instância do wrapper, passando dados escalados
                # Assumindo que o método 'treinar' retorna a própria instância treinada ou o modelo interno
                # Ajuste se seu método 'treinar' tiver outra assinatura
                modelo_resultante = modelo_wrapper_instancia.treinar(X_scaled, y)

                # Verifica se o treinamento foi bem-sucedido (depende do que 'treinar' retorna)
                # Se 'treinar' retorna self, a instância já está treinada.
                # Se 'treinar' retorna o modelo sklearn, precisamos garantir que o wrapper o armazene.
                # Vamos assumir que a instância do wrapper é o que queremos armazenar.
                if modelo_wrapper_instancia.modelo is not None: # Verifica se o modelo interno foi treinado
                    print(f"  ✅ Modelo {nome} treinado com sucesso")
                    # Adiciona a *instância do wrapper* treinada ao pool
                    modelos_treinados.append(modelo_wrapper_instancia)
                else:
                    print(f"  ❌ Falha ao treinar modelo {nome} (modelo interno é None)")

            except Exception as e:
                print(f"  ❌ Erro ao treinar modelo {nome}: {e}")
                # traceback.print_exc() # Descomente para debug detalhado

        # Fallback se nenhum modelo foi treinado
        if not modelos_treinados:
            print("\n⚠️ AVISO: Nenhum modelo foi treinado com sucesso. Tentando Regressão Linear como fallback...")
            try:
                modelo_fallback_wrapper = LinearRegressionModelo()
                modelo_fallback_wrapper.treinar(X_scaled, y) # Treina com dados escalados
                if modelo_fallback_wrapper.modelo is not None:
                    modelos_treinados.append(modelo_fallback_wrapper)
                    print("  ✅ Modelo de fallback (Regressão Linear) treinado com sucesso.")
                else:
                     print("  ❌ Erro crítico: Não foi possível treinar o modelo de fallback.")
            except Exception as e:
                print(f"  ❌ Erro crítico ao treinar fallback: {e}")

        print(f"\n✓ Treinamento inicial concluído. Modelos no pool: {len(modelos_treinados)}")
        return modelos_treinados, scaler

    @staticmethod
    def selecionar_melhor_modelo(pool_modelos, janela_dados, scaler):
        """
        Seleciona o melhor modelo do pool com base no MSE na janela de dados recentes.

        Args:
            pool_modelos (list): Lista de modelos wrapper no pool.
            janela_dados (list): Lista de tuplas (x, y) recentes.
            scaler (StandardScaler): O scaler ajustado nos dados iniciais.

        Returns:
            object: A instância do modelo wrapper com melhor desempenho, ou None.
        """
        if not pool_modelos:
            print("⚠️ AVISO: Pool de modelos está vazio!")
            return None
        if not janela_dados:
            print("⚠️ AVISO: Janela de dados está vazia!")
            # Retorna o último modelo adicionado ou o primeiro como fallback
            return pool_modelos[-1] if pool_modelos else None

        X_janela_list = [x for x, y in janela_dados]
        y_janela = np.array([y for x, y in janela_dados])

        # Tenta escalar os dados da janela
        try:
            # Transforma os dados da janela usando o scaler original
            X_janela_scaled = scaler.transform(np.array(X_janela_list))
        except Exception as e:
            print(f"⚠️ AVISO: Erro ao escalar dados da janela: {e}. Usando dados não escalados.")
            X_janela_scaled = np.array(X_janela_list) # Fallback para dados não escalados

        melhor_modelo = None
        menor_erro = float('inf')

        print(f"  Avaliando {len(pool_modelos)} modelos do pool na janela ({len(y_janela)} amostras)...")
        for i, modelo_wrapper in enumerate(pool_modelos):
            try:
                # Usa o método 'prever' do wrapper, que deve lidar com o modelo interno
                # Passa os dados da janela escalados
                y_pred = modelo_wrapper.prever(X_janela_scaled)
                erro = mean_squared_error(y_janela, y_pred)
                # print(f"    Modelo {i} ({modelo_wrapper.nome if hasattr(modelo_wrapper, 'nome') else type(modelo_wrapper).__name__}): MSE = {erro:.4f}") # Debug

                if erro < menor_erro:
                    menor_erro = erro
                    melhor_modelo = modelo_wrapper

            except Exception as e:
                print(f"  ❌ Erro ao avaliar modelo #{i} ({type(modelo_wrapper).__name__}): {e}")

        if melhor_modelo is None:
            print("  ERRO: Nenhum modelo pôde ser avaliado! Retornando o último modelo do pool.")
            return pool_modelos[-1] if pool_modelos else None
        else:
            print(f"  ✓ Melhor modelo selecionado: {melhor_modelo.nome if hasattr(melhor_modelo, 'nome') else type(melhor_modelo).__name__} (MSE: {menor_erro:.4f})")
            return melhor_modelo

    @staticmethod
    def desempenho(modelo_wrapper, janela_dados, scaler):
        """
        Avalia o desempenho (MSE) de um modelo na janela de dados recentes.

        Args:
            modelo_wrapper (object): A instância do modelo wrapper a ser avaliado.
            janela_dados (list): Lista de tuplas (x, y) recentes.
            scaler (StandardScaler): O scaler ajustado nos dados iniciais.

        Returns:
            float: O erro quadrático médio (MSE), ou float('inf') em caso de erro.
        """
        if not janela_dados:
            return float('inf') # Não pode avaliar em janela vazia

        X_janela_list = [x for x, y in janela_dados]
        y_janela = np.array([y for x, y in janela_dados])

        try:
            X_janela_scaled = scaler.transform(np.array(X_janela_list))
            y_pred = modelo_wrapper.prever(X_janela_scaled)
            return mean_squared_error(y_janela, y_pred)
        except Exception as e:
            print(f"  ❌ Erro ao calcular desempenho para {type(modelo_wrapper).__name__}: {e}")
            return float('inf') # Retorna erro infinito se a avaliação falhar

    @staticmethod
    def adicionar_a_janela(janela, dado, tamanho_max=100):
        """Adiciona um novo dado à janela, mantendo o tamanho máximo."""
        janela.append(dado)
        if len(janela) > tamanho_max:
            return janela[1:] # Retorna uma nova lista sem o elemento mais antigo
        return janela

    @staticmethod
    def get_state(detector_wrapper):
        """
        Obtém o estado padronizado ('NORMAL', 'ALERTA', 'MUDANÇA') do detector.
        Assume que o wrapper tem a property 'drift_detectado' e um atributo 'detector'
        que armazena a instância do detector 'river'.

        Args:
            detector_wrapper: A instância do wrapper do detector (e.g., KSWINDetector).

        Returns:
            str: O estado atual ('NORMAL', 'ALERTA', 'MUDANÇA').
        """
        try:
            # 1. Verificar MUDANÇA (usa a property 'drift_detectado' do wrapper)
            if hasattr(detector_wrapper, 'drift_detectado') and detector_wrapper.drift_detectado:
                return "MUDANÇA"

            # 2. Verificar ALERTA (acessa o detector 'river' interno)
            river_detector = getattr(detector_wrapper, 'detector', None)
            if river_detector and hasattr(river_detector, 'in_warning_zone') and river_detector.in_warning_zone:
                 # Apenas DDM e HDDM_W (do river.drift.binary) têm 'in_warning_zone'
                 return "ALERTA"

            # 3. Se não houver mudança nem alerta, estado é NORMAL
            return "NORMAL"

        except Exception as e:
            print(f"⚠️ Erro ao obter estado do detector {type(detector_wrapper).__name__}: {e}")
            return "NORMAL" # Fallback seguro