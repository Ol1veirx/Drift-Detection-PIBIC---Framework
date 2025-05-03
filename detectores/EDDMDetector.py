from detectores.DetectorDriftBase import DetectorDriftBase
from river.drift.binary import EDDM

class EDDMDetector(DetectorDriftBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.detector = EDDM(**kwargs)

    def atualizar(self, erro):
        self.detector.update(erro)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected