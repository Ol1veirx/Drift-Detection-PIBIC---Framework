from frameworkClasses.prediction_strategy import PredictionStrategy


class StandardPredictionStrategy(PredictionStrategy):
    """Estratégia de previsão padrão usando ensemble"""

    def predict(self, processor, x_scaled, estado):
        from frameworkDetector.framework_detector import FrameworkDetector

        if estado == "NORMAL":
            return processor.modelo_atual.prever(x_scaled)[0]

        pool_atual = processor.pools_por_regime.get(processor.regime_atual, [])
        if len(pool_atual) <= 1:
            return processor.modelo_atual.prever(x_scaled)[0]

        erros_modelos = []
        for modelo in pool_atual:
            erro = FrameworkDetector.desempenho(modelo, processor.janela_dados_recentes, processor.scaler)
            erros_modelos.append((modelo, erro))

        erros_modelos.sort(key=lambda x: x[1])
        top_modelos = erros_modelos[:min(3, len(erros_modelos))]

        pesos = [1/(erro+0.001) for _, erro in top_modelos]
        soma_pesos = sum(pesos)
        pesos_normalizados = [p/soma_pesos for p in pesos]

        predicao = 0
        for (modelo, _), peso in zip(top_modelos, pesos_normalizados):
            predicao += modelo.prever(x_scaled)[0] * peso

        return predicao