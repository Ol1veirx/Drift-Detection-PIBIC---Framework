from abc import ABC, abstractmethod
from avaliacao.DriftEvaluator import DriftEvaluator

class AvaliadorDriftBase(ABC):
    @abstractmethod
    def prequential(self, X, Y, tamanho_batch, modelo_classe, detector_classe=None):
        pass

    def executar_avaliacao(self, X, Y, tamanho_batch, modelo_classe, detector_classe=None):
        return self.prequential(X, Y, tamanho_batch, modelo_classe, detector_classe)


class AvaliadorBatch(AvaliadorDriftBase):
    def prequential(self, X, Y, tamanho_batch, modelo_classe, detector_classe):
        return DriftEvaluator.prequential_batch(X, Y, tamanho_batch, modelo_classe, detector_classe)


class AvaliadorPassivo(AvaliadorDriftBase):
    def prequential(self, X, Y, tamanho_batch, modelo_classe, detector_classe=None):
        return DriftEvaluator.prequential_passivo(X, Y, tamanho_batch, modelo_classe)