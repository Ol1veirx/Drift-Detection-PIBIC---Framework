import numpy as np
from preprocessamento.SeriesProcessor import SeriesProcessor

class Experimento:
    def __init__(self, series, modelos, tamanho_batch=100, lags=5, repeticoes=3):
        self.series = series
        self.modelos = modelos  # Lista de dicionários com {"nome", "avaliador", "modelo", "detector"}
        self.tamanho_batch = tamanho_batch
        self.lags = lags
        self.repeticoes = repeticoes

    def preprocessar_serie(self, nome_serie):
        """
        Baixa, normaliza e transforma uma série temporal em janelas X e Y.
        """

        serie = SeriesProcessor.baixar_dados(nome_serie)
        serie = SeriesProcessor.normalizar_serie(serie)
        X, Y = SeriesProcessor.criar_janela_temporal(serie, self.lags)

        return X, Y

    def executar(self):
        """
        Executa cada modelo N vezes para cada uma das séries
        """
        # variavel para salvar os resultados
        resultados = []

        # iterar sobre a quantidade de séries passadas
        for nome_serie in self.series:

            # processando a serie antes de usar
            X, Y = self.preprocessar_serie(nome_serie)

            # rodando modelo por modelo
            for modelo_cfg in self.modelos:

                # configurando a saida dos modelos
                nome_modelo = modelo_cfg["nome"]
                avaliador = modelo_cfg["avaliador"]
                modelo = modelo_cfg["modelo"]
                detector = modelo_cfg.get("detector")  # pode ser None

                print(f"Executando {nome_modelo} na série: {nome_serie}")

                # rodando cada modelo N vezes
                for repeticao in range(self.repeticoes):
                    if detector:
                        _, detecs, mae = avaliador.executar_avaliacao(X, Y, self.tamanho_batch, modelo, detector)
                        qtd_deteccoes = len(detecs)
                    else:
                        _, mae = avaliador.executar_avaliacao(X, Y, self.tamanho_batch, modelo)
                        qtd_deteccoes = None

                    # formatando a linha para ser escrita
                    resultados.append({
                        "serie": nome_serie,
                        "modelo": nome_modelo,
                        "repeticao": repeticao + 1,
                        "mae": float(np.ravel(mae)[0]),
                        "qtd_deteccoes": qtd_deteccoes
                    })

        return resultados
