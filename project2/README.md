# Projeto Extração de Grafo de Conhecimento Clínico do MultiCaRe com N-gramas e Aprendizado de Máquina

## Project MultiCaRe Clinical Knowledge Graph Extraction with N-grams and Machine Learning

## Slides
- [Apresentação](https://docs.google.com/presentation/d/1Uv2PO4QWB4y54Y-5D5rHIStLSTYr8B1M/edit?slide=id.p1#slide=id.p1)

## Demonstração Executável

- Comparação completa: notebook [`pipelines/notebooks/comparacao_modelos.ipynb`](pipelines/notebooks/comparacao_modelos.ipynb)
  (ou `python -m src.comparison.run_comparison`, ver [`src/README.md`](src/README.md)).
- Relatório gerado automaticamente: [`data/processed/comparison/comparison_report.md`](data/processed/comparison/comparison_report.md).
- Pipelines dos modelos: [`pipeline_ml_tradicional.ipynb`](pipelines/notebooks/pipeline_ml_tradicional.ipynb) e [`pipeline_rede_neural.ipynb`](pipelines/notebooks/pipeline_rede_neural.ipynb).

## Metodologia

O Projeto 2 substitui as regras do Projeto 1 por **aprendizado de máquina clássico e redes neurais sobre N-gramas**, sem LLMs, embeddings pré-treinados ou Transformers.

1. **Candidatos (`Preprocess.py`)**: limpeza do texto, segmentação em sentenças e **sintagmas nominais** como candidatos a entidade. Cada par de candidatos a até 10 tokens gera o **texto intermediário** (trecho entre as entidades) usado como contexto da relação.
2. **Entidades (`EntityClassifier`)**: lematização + TF-IDF de **1 a 5-gramas** + Regressão Logística. Classes: `PATIENT`, `SYMPTOM`, `FINDING`, `DIAGNOSIS`, `PROCEDURE`, `VALUE`, `NOT_ENTITY`.
3. **Relações (`RelationClassifier`)**: TF-IDF de 1 a 5-gramas do contexto + *one-hot* dos tipos + Regressão Logística. Classes: `HAS_SYMPTOM`, `HAS_VALUE`, `SUPPORTS`, `UNDERWENT_PROCEDURE`, `TREATED_BY`, `NO_RELATION`. Treinado com a **anotação manual** (`data/annotations/text01..12.csv`).
4. **Weak supervision (MLP)**: as arestas geradas pelas regras do Projeto 1 viram rótulos automáticos para um **MLP em PyTorch** (TF-IDF 1–2-gramas → 256 → 128 → classes).
5. **Sistema de comparação (`src/comparison`)**: avalia todos os modelos e o Projeto 1 num **gold standard** de 3 documentos (`gold01..03.csv`, 191 pares, fora de qualquer treino), com Precision/Recall/F1, matriz de confusão, IC 95% por *bootstrap*, ablação de n-gramas e análise do grafo resultante.

```python
# uso do sistema de comparação
from src.comparison import run_comparison
res = run_comparison.run(include_mlp=True, include_extras=True)
```

Na comparação, o Projeto 1 é lido da sua **saída oficial** (`nodes.csv`/`edges.csv`) com casamento relaxado de menções (sobreposição de tokens ≥ 0,5), o que o favorece. Modelos `[manual]` recebem os tipos de entidade do gold; a variante `tipos previstos` usa o `EntityClassifier`, para uma comparação sem oráculo.

## Confronto com Trabalho Anterior

O Projeto 1 usa regras (verbos-gatilho, dicionários de palavras-chave, distância em caracteres). O Projeto 2 aprende esses padrões dos exemplos. O confronto é **experimental**, no mesmo gold, e está detalhado em
[`data/processed/comparison/comparison_report.md`](data/processed/comparison/comparison_report.md).

TODO (preencher com os números do relatório após rodar): macro-F1 do Projeto 1 × melhor modelo de ML × MLP com weak supervision; recall de detecção de relações do Projeto 1; efeito da supervisão manual × weak supervision; efeito dos n-gramas; grafos em excesso/perdidos.

![Ranking](assets/images/comparison/01_ranking_modelos_relacoes.png)

## Trabalhos Estudados

* Nievas Offidani et al. (2025): conjunto de dados MultiCaRe, origem dos textos.
* Ji et al. (2022): *A Survey on Knowledge Graphs* — representação, aquisição e aplicações.
* TODO: incluir as referências adicionais pesquisadas pela equipe (PMI, CRF, BiLSTM-CRF etc.).

## Modelo Lógico

Grafo de propriedades: nós `PATIENT`, `SYMPTOM`, `FINDING`, `DIAGNOSIS`, `PROCEDURE`, `VALUE` (propriedades: `label_original`, `label_normalizado`, `span_start`, `span_end`, `value`, `unit`); arestas `HAS_SYMPTOM`, `HAS_VALUE`, `SUPPORTS`, `UNDERWENT_PROCEDURE`, `TREATED_BY` (propriedade: `intermediate_text`).

![Esquema dos grafos](assets/images/comparison/08_esquema_dos_grafos.png)

## Análises que podem ser realizadas

* Quais relações cada método perde ou produz em excesso (`grafos_comparacao.csv`).
* Trajetória clínica paciente → sintoma → exame → tratamento.
* Associações entre achados e valores (`HAS_VALUE`) por tipo de exame.
* Quais N-gramas do contexto mais pesam em cada relação (coeficientes da Regressão Logística).

## Ferramentas

Python, pandas, scikit-learn (TF-IDF, LR, NB, SVM, RF), PyTorch (MLP), spaCy/scispaCy (lematização e sintagmas nominais), matplotlib (gráficos), PyVis (grafo interativo do Projeto 1).

## Resultados

Gráficos gerados em `assets/images/comparison/` e tabelas em `data/processed/comparison/`:

![Precisão x Recall](assets/images/comparison/02_precisao_vs_recall.png)
![F1 por classe](assets/images/comparison/03_f1_por_classe.png)
![Matrizes de confusão](assets/images/comparison/04_matrizes_confusao.png)
![Ablação de n-gramas](assets/images/comparison/05_ablacao_ngramas.png)
![Entidades](assets/images/comparison/06_entidades_f1_por_tipo.png)
![Distribuição de relações](assets/images/comparison/07_distribuicao_relacoes_grafo.png)

**Ressalvas**: gold pequeno (3 documentos; ICs largos), gold não exaustivo (precisão ponta a ponta do Projeto 1 subestimada) e comparação de relações sobre pares candidatos do gold. TODO: discussão dos resultados.

## Como Modelos de Linguagem foram Usados

O pipeline de extração **não** usa LLMs, embeddings pré-treinados nem Transformers. Um LLM (Claude) foi usado como **assistente de desenvolvimento** no sistema de comparação (`src/comparison`, testes, gráficos e esta documentação); o código foi revisado e executado pela equipe. TODO: os demais membros listarem o uso nas suas partes.

## Referências Bibliográficas

* Nievas Offidani, M., Roffet, F., González Galtier, M. C., Massiris, M., & Delrieux, C. (2025). An Open-Source Clinical Case Dataset for Medical Image Classification and Multimodal AI Applications. *Data*, 10(8), 123. https://doi.org/10.3390/DATA10080123
* Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2022). A Survey on Knowledge Graphs: Representation, Acquisition, and Applications. *IEEE TNNLS*, 33(2), 494–514. https://doi.org/10.1109/TNNLS.2021.3070843
