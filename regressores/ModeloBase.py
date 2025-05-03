from abc import ABC, abstractmethod

class ModeloBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def treinar(self, X, y):
        pass

    @abstractmethod
    def prever(self, X):
        pass