from abc import ABC, abstractmethod


class PredictionStrategy(ABC):
    """Estratégia base para previsões"""

    @abstractmethod
    def predict(self, processor, x_scaled, estado):
        pass