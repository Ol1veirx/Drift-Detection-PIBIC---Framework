from detectores.DetectorDriftBase import DetectorDriftBase
from river.drift.binary import DDM

class DDMDetector(DetectorDriftBase):
    def __init__(self, threshold=0.05, **kwargs):
        super().__init__()
        self.detector = DDM(**kwargs)
        self.threshold = threshold

    def atualizar(self, erro):
        erro_binario = 1 if erro > self.threshold else 0
        self.detector.update(erro_binario)

    @property
    def drift_detectado(self):
        return self.detector.drift_detected