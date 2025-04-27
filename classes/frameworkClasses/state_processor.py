from abc import ABC, abstractmethod


class StateProcessor(ABC):
    """Classe base para processadores de estado"""

    @abstractmethod
    def process(self, processor, indice_global, x_t, y_t, x_t_scaled, i, detector_escolhido):
        pass