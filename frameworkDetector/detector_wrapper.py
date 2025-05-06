class DetectorWrapper:
    """
    Wrapper para padronizar a interface com qualquer detector de drift.
    Esta classe permite usar qualquer detector externo sem modificá-lo.
    """

    def __init__(self, detector):
        """
        Inicializa o wrapper com um detector.

        Parâmetros:
        detector: Objeto detector de drift que tenha método de atualização
        """
        self.detector = detector

    def atualizar(self, valor):
        """
        Atualiza o detector com um novo valor.
        Adapta para diferentes interfaces dos detectores.
        """
        # Tenta chamar método de atualização do detector
        # (compatível com a maioria dos detectores)
        if hasattr(self.detector, 'update'):
            self.detector.update(valor)
        elif hasattr(self.detector, 'add_element'):
            self.detector.add_element(valor)
        elif hasattr(self.detector, 'atualizar'):
            self.detector.atualizar(valor)
        else:
            raise AttributeError("Detector não possui método de atualização compatível")

    @property
    def drift_detectado(self):
        """Verifica se o detector identificou um drift."""
        # Adaptação para diferentes interfaces de detectores
        if hasattr(self.detector, 'drift_detectado'):
            return self.detector.drift_detectado
        elif hasattr(self.detector, 'detected_change'):
            return self.detector.detected_change()
        elif hasattr(self.detector, 'detected_warning_zone'):
            # Alguns detectores têm zona de alerta e mudança
            return self.detector.detected_warning_zone()
        else:
            # Se não encontrar nenhum padrão conhecido, assume sem drift
            return False