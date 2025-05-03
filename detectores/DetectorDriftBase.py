from abc import ABC, abstractmethod

class DetectorDriftBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def atualizar(self, erro):
        pass

    @property
    @abstractmethod
    def drift_detectado(self):
        pass