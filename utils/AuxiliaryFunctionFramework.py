import copy
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from classes.frameworkDetector.framework_detector import FrameworkDetector


class AuxiliaryFunctionFramework:

    @staticmethod
    def get_model_name(model_wrapper):
        """Retorna o nome do modelo wrapper ou o nome da classe."""
        return model_wrapper.nome if hasattr(model_wrapper, 'nome') else type(model_wrapper).__name__

    @staticmethod
    def perform_prediction(model, scaler, x_t, fallback_value):
        """Realiza a predição com tratamento de erro."""
        try:
            x_t_reshaped = x_t.reshape(1, -1)
            x_t_scaled = scaler.transform(x_t_reshaped)
            return model.prever(x_t_scaled)[0], x_t_scaled
        except Exception as e:
            print(f"\n⚠️ Erro na predição: {e}")
            print(f"  Usando predição de fallback: {fallback_value:.4f}")
            return fallback_value, None
    @staticmethod
    def update_detector(detector, error_value):
        """Atualiza o detector com tratamento de erro."""
        try:
            detector.atualizar(error_value)
        except Exception as e:
            print(f"\n⚠️ Erro ao atualizar detector: {e}")

    @staticmethod
    def get_detector_state(detector):
        """Obtém o estado do detector com tratamento de erro."""
        try:
            return FrameworkDetector.get_state(detector)
        except Exception as e:
            print(f"\n⚠️ Erro ao obter estado do detector: {e}")
            return "NORMAL"

    @staticmethod
    def handle_alert_state(model, x_t_scaled, y_t):
        """Executa ações para o estado de ALERTA (ex: partial_fit)."""
        if hasattr(model, "partial_fit") and x_t_scaled is not None:
            try:
                model.partial_fit(x_t_scaled, np.array([y_t]))
            except Exception as e:
                # print(f"  (Info) Erro no partial_fit em ALERTA: {e}") # Opcional
                pass

    @staticmethod
    def add_model_to_pool(pool, model_to_add):
        """Adiciona uma cópia do modelo ao pool com tratamento de erro."""
        try:
            model_copy = copy.deepcopy(model_to_add)
            pool.append(model_copy)
            print(f"  Adicionando cópia de '{AuxiliaryFunctionFramework.get_model_name(model_copy)}' ao pool.")
            print(f"  Tamanho do pool agora: {len(pool)}")
        except Exception as e:
            print(f"  ⚠️ Erro ao adicionar modelo ao pool: {e}")

    @staticmethod
    def select_and_evaluate_best_model(pool, window, scaler, threshold):
        """Seleciona e avalia o melhor modelo do pool."""
        print("  Selecionando melhor modelo do pool...")
        best_model = None
        mse_best = float('inf')
        try:
            best_model = FrameworkDetector.selecionar_melhor_modelo(pool, window, scaler)
            if best_model:
                model_name = AuxiliaryFunctionFramework.get_model_name(best_model)
                print(f"  Avaliando desempenho de '{model_name}'...")
                try:
                    mse_best = FrameworkDetector.desempenho(best_model, window, scaler)
                    print(f"  MSE na janela: {mse_best:.4f}. Limiar: {threshold:.4f}")
                except Exception as e_eval:
                    print(f"  ⚠️ Erro ao calcular desempenho do modelo '{model_name}': {e_eval}")
                    mse_best = float('inf') # Considera desempenho ruim se houve erro
            else:
                print("  Nenhum modelo selecionado do pool.")
        except Exception as e_select:
            print(f"  ⚠️ Erro durante a seleção do melhor modelo: {e_select}")

        return best_model, mse_best

    @staticmethod
    def retrain_model(model_type_to_train, window, scaler, min_samples):
        """Tenta retreinar um novo modelo do tipo especificado."""
        print(f"    Instanciando novo modelo do tipo: {model_type_to_train.__name__}")
        new_model_wrapper = model_type_to_train()

        if not window or not isinstance(window[0], tuple) or len(window[0]) != 2:
             print("  ❌ Erro: Formato inválido da janela para retreino.")
             return None

        X_recent_list = [x for x, _ in window]
        y_recent = np.array([y for _, y in window])

        if len(X_recent_list) < min_samples:
            print(f"  ⚠️ Janela com poucos dados ({len(X_recent_list)} < {min_samples}), não é possível retreinar.")
            return None

        try:
            X_recent_scaled = scaler.transform(np.array(X_recent_list))
            new_model_wrapper.treinar(X_recent_scaled, y_recent)
            if new_model_wrapper.modelo is not None:
                print(f"  ✓ Novo modelo ({AuxiliaryFunctionFramework.get_model_name(new_model_wrapper)}) treinado.")
                return new_model_wrapper
            else:
                print("  ❌ Falha ao treinar novo modelo (modelo interno é None).")
                return None
        except Exception as e:
            print(f"  ❌ Erro durante o retreino: {e}.")
            return None

    @staticmethod
    def determine_next_model(current_model, pool, window, scaler, threshold, min_samples_retrain):
        """
        Lógica para decidir qual modelo usar após detectar uma mudança.
        Prioriza: Melhor do Pool (se bom) > Retreinamento > Melhor do Pool (fallback) > Modelo Atual.
        """
        AuxiliaryFunctionFramework.add_model_to_pool(pool, current_model)
        best_pool_model, mse_best = AuxiliaryFunctionFramework.select_and_evaluate_best_model(pool, window, scaler, threshold)

        # 1. Tentar usar o melhor modelo do pool se for bom o suficiente
        if best_pool_model and mse_best < threshold:
            print(f"  ✓ Ativando modelo do pool: {AuxiliaryFunctionFramework.get_model_name(best_pool_model)}")
            return best_pool_model

        # 2. Se o modelo do pool não for bom ou não existir, tentar retreinar
        if best_pool_model:
            print(f"  Desempenho do melhor modelo do pool não satisfatório ({mse_best:.4f} >= {threshold:.4f}). Tentando retreinar...")
        else:
            print("  Nenhum modelo satisfatório no pool. Tentando retreinar...")

        model_type_to_retrain = type(best_pool_model) if best_pool_model else type(current_model)
        retrained_model = AuxiliaryFunctionFramework.retrain_model(model_type_to_retrain, window, scaler, min_samples_retrain)

        if retrained_model:
            print(f"  ✓ Novo modelo retreinado ativado: {AuxiliaryFunctionFramework.get_model_name(retrained_model)}")
            return retrained_model

        # 3. Se retreino falhou, usar o melhor do pool como fallback (se existir)
        if best_pool_model:
            print("  ❌ Retreino falhou. Usando melhor modelo do pool como fallback.")
            return best_pool_model

        # 4. Se tudo falhou, manter o modelo atual
        print("  ❌ Retreino falhou e não há modelo de fallback no pool. Mantendo modelo anterior.")
        return current_model

    @staticmethod
    def handle_change_state(current_model, pool, window, scaler, threshold, min_samples_retrain, k_reset):
        """
        Executa ações para o estado de MUDANÇA: determina o próximo modelo e ajusta a janela.
        Retorna: tuple (novo_modelo_ativo, nova_janela_modificada)
        """
        local_window = list(window) # Cria cópia para modificar

        # Determina qual será o próximo modelo ativo
        active_model = AuxiliaryFunctionFramework.determine_next_model(
            current_model, pool, local_window, scaler, threshold, min_samples_retrain
        )

        # --- Reset Parcial da Janela ---
        if len(local_window) > k_reset:
            local_window = local_window[-k_reset:]

        return active_model, local_window

    @staticmethod
    def calculate_and_store_metrics(model, scaler, window, index, min_samples,
                                    metricas_rmse_stream, metricas_mae_stream, metricas_r2_stream):
        """Calcula e armazena métricas de desempenho periodicamente."""
        if len(window) <= min_samples: # Verifica se há dados suficientes
             # print(f"  (Info) Janela com poucos dados ({len(window)} <= {min_samples}) para métricas no índice {index}.") # Opcional
             return

        if not isinstance(window[0], tuple) or len(window[0]) != 2:
            print(f"  (Info) Formato inválido da janela para métricas no índice {index}.")
            return

        try:
            X_eval = np.array([x for x, _ in window])
            y_eval = np.array([y for _, y in window])
            X_eval_scaled = scaler.transform(X_eval)
            y_pred_eval = model.prever(X_eval_scaled)

            rmse = np.sqrt(mean_squared_error(y_eval, y_pred_eval))
            mae = mean_absolute_error(y_eval, y_pred_eval)
            r2 = r2_score(y_eval, y_pred_eval)

            metricas_rmse_stream.append((index, rmse))
            metricas_mae_stream.append((index, mae))
            metricas_r2_stream.append((index, r2))
        except Exception as e:
            print(f"  (Info) Erro ao calcular métricas no índice {index}: {e}")
            pass # Evita parar o processo por erro no cálculo de métricas

    @staticmethod
    def update_window(window, data_point, max_size):
        """Adiciona novo ponto à janela com tratamento de erro."""
        try:
            return FrameworkDetector.adicionar_a_janela(window, data_point, tamanho_max=max_size)
        except Exception as e:
            print(f"\n⚠️ Erro ao adicionar dado à janela: {e}")
            return window