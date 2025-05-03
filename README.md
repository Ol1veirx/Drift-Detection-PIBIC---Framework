# Framework de Detecção de Drift com Gerenciamento de Regimes

## Conceito de Regimes

No contexto deste framework, **regimes** referem-se a diferentes padrões ou comportamentos estatísticos que podem surgir sequencialmente em um fluxo de dados (stream). Um fluxo de dados pode passar por períodos onde suas características subjacentes (como média, variância ou relações entre variáveis) permanecem relativamente estáveis (um regime) e, em seguida, mudar para um novo conjunto de características (um novo regime). Essas mudanças são frequentemente associadas a *concept drift*.

A ideia é que um único modelo preditivo pode não ser ótimo para todos os regimes. Ao identificar o regime atual, o framework pode selecionar ou treinar um modelo mais adequado para aquele comportamento específico dos dados, melhorando a precisão geral.

## Ideia Principal do Framework

Este framework foi projetado para processar fluxos de dados contínuos, detectar *concept drifts* e adaptar-se a eles de forma inteligente, gerenciando múltiplos modelos organizados por regimes de dados identificados.

O fluxo principal é:

1.  **Processamento Contínuo:** Processa o stream de dados amostra por amostra.
2.  **Detecção de Drift:** Utiliza um detector de drift (ex: KSWIN, ADWIN) para monitorar o erro de previsão ou outra métrica e sinalizar quando ocorrem mudanças significativas (drifts) ou avisos (warnings).
3.  **Identificação de Regime:** Periodicamente, analisa características da janela de dados mais recente para determinar a qual regime os dados atuais pertencem (usando clustering, como visto em [`FrameworkDetector.identificar_regime`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkDetector/framework_detector.py)).
4.  **Gerenciamento de Pools de Modelos:** Mantém pools separados de modelos preditivos, um para cada regime identificado ([`StreamProcessor.pools_por_regime`](#file:StreamProcessor.py)).
5.  **Adaptação:**
    *   **Estado Normal/Alerta:** Faz previsões usando o modelo atual ou uma combinação de modelos do pool do regime atual. Periodicamente, adiciona cópias do modelo atual ao pool do regime correspondente ([`NormalStateProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/normal_state_processor.py), [`AlertStateProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/alert_state_processor.py)). Em alerta, pode tentar modelos alternativos do pool.
    *   **Estado de Mudança (Drift):** Ao detectar um drift, avalia os modelos no pool do regime atual, seleciona o melhor ([`FrameworkDetector.selecionar_melhor_modelo`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkDetector/framework_detector.py)), reconfigura o pool ([`FrameworkDetector.gerenciar_pool`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkDetector/framework_detector.py)) e inicia a coleta de dados para treinar um novo modelo ([`ChangeStateProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/change_state_processor.py)).
    *   **Coleta Pós-Drift:** Coleta um número definido de amostras do novo conceito ([`DriftCollectionProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/drift_collection_processor.py)).
    *   **Retreinamento:** Treina um novo modelo com os dados coletados ([`FrameworkDetector.treinar_novo_conceito`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkDetector/framework_detector.py)) e o adiciona ao pool do regime atual ([`FrameworkDetector.adicionar_ao_pool`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkDetector/framework_detector.py)).
6.  **Seleção de Modelo:** Em caso de degradação de desempenho detectada periodicamente, o framework avalia modelos de diferentes pools de regime e pode trocar para um modelo de outro regime se ele apresentar melhor desempenho nos dados recentes ([`StreamProcessor._calcular_metricas_periodicas`](#file:StreamProcessor.py)).

O objetivo final é manter a performance preditiva alta ao longo do tempo, adaptando-se às mudanças nos dados através da detecção de drift e do gerenciamento de modelos especializados por regime.

## Mapeamento Resumido das Classes

*   **`frameworkDetector/framework_detector.py` ([`FrameworkDetector`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkDetector/framework_detector.py)):** Classe utilitária com métodos estáticos centrais para treinar modelos (inicial, novo conceito), selecionar o melhor modelo de um pool, avaliar desempenho, adicionar/gerenciar pools, identificar regimes e obter o estado do detector.
*   **`utils/StreamProcessor.py` ([`StreamProcessor`](#file:StreamProcessor.py)):** Orquestrador principal do framework. Gerencia o ciclo de vida do processamento do stream, coordena a detecção, identificação de regime, seleção de estratégias de estado e previsão, e coleta de resultados.
*   **`frameworkClasses/`:** Contém classes que definem comportamentos específicos:
    *   **`state_processor.py` ([`StateProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/state_processor.py)):** Classe base abstrata para processadores de estado.
    *   **`normal_state_processor.py` ([`NormalStateProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/normal_state_processor.py)):** Lógica para o estado 'NORMAL'.
    *   **`alert_state_processor.py` ([`AlertStateProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/alert_state_processor.py)):** Lógica para o estado 'ALERTA'.
    *   **`change_state_processor.py` ([`ChangeStateProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/change_state_processor.py)):** Lógica para o estado 'MUDANÇA' (drift detectado).
    *   **`drift_collection_processor.py` ([`DriftCollectionProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/drift_collection_processor.py)):** Lógica para a fase de coleta de dados pós-drift.
    *   **`prediction_strategy.py` ([`PredictionStrategy`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/prediction_strategy.py)):** Classe base abstrata para estratégias de previsão.
    *   **`standard_prediction_strategy.py` ([`StandardPredictionStrategy`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/standard_prediction_strategy.py)):** Estratégia de previsão para estados NORMAL/ALERTA.
    *   **`drift_prediction_strategy.py` ([`DriftPredictionStrategy`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/drift_prediction_strategy.py)):** Estratégia de previsão durante a coleta pós-drift.
    *   **`modelo_transicao.py` ([`ModeloTransicao`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/modelo_transicao.py)):** Wrapper para um modelo simples usado durante transições.
    *   **`avaliador_framework.py` ([`AvaliadorFramework`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/frameworkClasses/avaliador_framework.py)):** Classe adaptadora para executar e avaliar o `StreamProcessor` dentro da estrutura de `Experimento`.
*   **`detectores/`:** Contém implementações de diferentes algoritmos de detecção de drift (ex: [`KSWINDetector`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/detectores/KSWINDetector.py), [`ADWINDetector`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/detectores/ADWINDetector.py)).
*   **`regressores/`:** Contém wrappers para modelos de regressão (offline e online), padronizando a interface com métodos `treinar` e `prever` (ex: [`RandomForestModelo`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/regressores/modelosOffline/RandomForestModelo.py)).
*   **`avaliacao/`:** Classes para avaliação de desempenho (ex: [`AvaliadorBatch`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/avaliacao/AvaliadorDriftBase.py)).
*   **`experimento/`:** Classe para orquestrar a execução de múltiplos testes comparativos ([`Experimento`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/experimento/ExperimentoDrift.py)).
*   **`preprocessamento/`:** Utilitários para processamento de séries temporais ([`SeriesProcessor`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/preprocessamento/SeriesProcessor.py)).
*   **`utils/`:** Outros utilitários como visualização ([`Visualizer`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/utils/Visualizer.py)) e gerenciamento de arquivos ([`FileManager`](/media/liedson/DADOS1/PIBIC/Drift-Detection-PIBIC---Framework/utils/FileManager.py)).
*   **Notebooks (`.ipynb`):**
    *   [`framework_demonstration.ipynb`](#framework_demonstration.ipynb): Demonstra o uso do framework passo a passo.
    *   [`framework_experiments.ipynb`](#framework_experiments.ipynb): Executa experimentos comparativos entre o framework (com diferentes configurações) e outras abordagens baseline.
