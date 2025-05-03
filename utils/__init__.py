"""
Pacote de utilitários para processamento de séries temporais, treinamento de modelos,
avaliação de drift e visualização de resultados.
"""

# Importando apenas os módulos locais do pacote utils
from .ModelTrainer import ModelTrainer
from .Visualizer import Visualizer

# Não tente importar módulos de outros pacotes aqui
# Remova estas importações:
# from ..preprocessamento.SeriesProcessor import SeriesProcessor
# from ..avaliacao.DriftEvaluator import DriftEvaluator
# from ..otimizador.Optimizer import Optimizer

__all__ = ['ModelTrainer', 'Visualizer']