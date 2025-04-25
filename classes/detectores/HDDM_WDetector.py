from classes.superclasse.DetectorDriftBase import DetectorDriftBase
from river.drift.binary import HDDM_W


class HDDM_WDetector(DetectorDriftBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.detector = HDDM_W(**kwargs)

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected