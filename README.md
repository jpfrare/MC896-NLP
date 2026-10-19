# Projeto Extração de Relações Clínicas com ML
## Project Clinical Relation Extraction with ML

### 1. Slides
[Link para os slides da apresentação]

### 2. Metodologia
Neste projeto, investigamos a transição de um sistema puramente baseado em regras (Projeto 1) para abordagens baseadas em Aprendizado de Máquina focadas na classificação de relações clínicas. O pipeline foi desenvolvido para avaliar o impacto da representação vetorial do texto intermediário usando N-gramas.

A extração foi dividida nas seguintes abordagens:
1. **Projeto 1 (Regras):** Baseline determinístico focado no casamento de padrões em grafos.
2. **Regressão Logística com N-gramas (Manual):** Utiliza um `FeatureUnion` de TF-IDF com word n-grams (1, 3) e char n-grams (2, 5) para capturar padrões sintáticos e morfológicos das relações.
3. **MLP (PyTorch):** Uma rede neural feedforward com três camadas ocultas (256, 128, num_classes) e dropout de 0.3. A entrada é um vetor TF-IDF gerado a partir do texto intermediário entre entidades clínicas.

### 3. Confronto com Trabalho Anterior
O Projeto 1 baseava-se estritamente na ocorrência exata de padrões linguísticos e proximidade. Esta abordagem resulta em alta precisão, mas falha em recuperar relações expressas com vocabulário não previsto pelas regras (baixo recall). 

Com os modelos estatísticos (LR e MLP), substituímos as regras rígidas por pesos associados à coocorrência de N-gramas no contexto das menções clínicas. A avaliação comparativa foi restrita a 5 relações válidas (`HAS_SYMPTOM`, `HAS_VALUE`, `SUPPORTS`, `TREATED_BY`, `UNDERWENT_PROCEDURE`), tratando as abstenções e classes inválidas estritamente como erros de recall em relação ao padrão-ouro.

### 6. Trabalhos Estudados
* Jurafsky, D. (2026). *CS 124: From Languages to Information*. Stanford University. Utilizado como base teórica para as representações de N-gramas e extração de relações.

### 7. Modelo Lógico
O modelo segue a ontologia básica estabelecida, onde entidades clínicas formam os nós e as classificações textuais formam as arestas (ex: `Disease -[HAS_SYMPTOM]-> Symptom`).
*(Substitua pela imagem exportada do modelo, caso exista)*

### 8. Análises que podem ser realizadas
* Análise da matriz de confusão para identificar sobreposição semântica entre classes preditas pelo MLP e pela Regressão Logística.
* Análise de densidade do grafo: verificação da quantidade de arestas extraídas por aprendizado de máquina em contraste com a esparsidade do grafo puramente baseado em regras.

### 9. Ferramentas
* **Python 3 / Pandas:** Manipulação e alinhamento dos dados de entrada do padrão-ouro.
* **Scikit-Learn:** Extração de features (TF-IDF), modelos clássicos (LogisticRegression) e cálculo de métricas (F1-score, Precision, Recall).
* **PyTorch:** Implementação e inferência da rede neural Multilayer Perceptron (MLP).
* **Seaborn / Matplotlib:** Geração de mapas de calor e gráficos de barras para análise de desempenho.

### 10. Resultados
Os modelos treinados apresentaram as seguintes métricas globais sobre as 5 classes reais:

| Modelo               |   Macro-F1 |   Precision |    Recall |
|:---------------------|-----------:|------------:|----------:|
| Projeto 1 (Regras)   |   0.132949 |    0.6      | 0.0756979 |
| LR N-gramas (Manual) |   0.650614 |    0.662065 | 0.645236  |
| MLP (PyTorch)        |   0.453853 |    0.442074 | 0.473263  |

**Comparação de F1-Score por Classe:**
![F1 por Classe](assets/images/comparison/02_f1_por_classe.png)

**Matrizes de Confusão:**
![Matrizes de Confusão](assets/images/comparison/03_matrizes_confusao.png)

### 10. Como Modelos de Linguagem foram Usados
Modelos de linguagem (LLMs) não foram utilizados no pipeline de inferência ou na extração do grafo, respeitando as restrições metodológicas do projeto. 
Entretanto, o Gemini foi utilizado como "Dialectical Partner" durante o desenvolvimento de software para:
* Refatoração e padronização do código Python em módulos orientados a objetos.
* Geração do script orquestrador de métricas (`classification_report` e `confusion_matrix`).
* Sugestão e formatação de gráficos no Seaborn.
* Ajuste de hiperparâmetros básicos para a Regressão Logística.

### 11. Referências Bibliográficas
* Santanchè, A. (2026). *Material da disciplina de Processamento de Línguas Naturais (nlp-040, nlp-110)*. Instituto de Computação, UNICAMP.
