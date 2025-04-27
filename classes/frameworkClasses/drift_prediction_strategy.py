from classes.frameworkClasses.prediction_strategy import PredictionStrategy


class DriftPredictionStrategy(PredictionStrategy):
    """Estratégia de previsão durante drift usando modelo de transição"""

    def predict(self, processor, x_scaled, i):
        if not processor.modelo_transicao or i % 10 == 0:
            processor.modelo_transicao = processor._criar_modelo_transicao()

        if processor.modelo_transicao:
            pct_transicao = min(processor.contador_novo_conceito / processor.observacoes_novo_conceito, 0.7)
            pred_antigo = processor.modelo_atual.prever(x_scaled)[0]
            pred_novo = processor.modelo_transicao.prever(x_scaled)[0]
            return pred_antigo * (1 - pct_transicao) + pred_novo * pct_transicao
        else:
            return processor.modelo_atual.prever(x_scaled)[0]
