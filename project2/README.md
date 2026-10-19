# Pipeline de Extração e Comparação (Projeto 2)

Este diretório contém os scripts responsáveis por avaliar e comparar os modelos de extração de relações do Projeto 1 e Projeto 2.

## Estrutura do Módulo `comparison`

* `config.py`: Definição de constantes, caminhos de pastas e dicionário de cores.
* `data.py`: Funções para o carregamento do padrão-ouro e tratamento das anotações.
* `metrics.py`: Computação estruturada de métricas (Precision, Recall, F1).
* `models.py`: Implementação da Regressão Logística usando TF-IDF de n-gramas de palavras e caracteres.
* `mlp_model.py`: Classe de inferência baseada no checkpoint PyTorch treinado para 4 relações.
* `p1_baseline.py`: Algoritmo de casamento relaxado de padrões que carrega as saídas do Projeto 1.
* `plots.py`: Funções de visualização (Seaborn/Matplotlib) dos resultados.
* `graph_analysis.py`: Extração de características estruturais dos grafos.
* `run_comparison.py`: Orquestrador principal.

## Como Executar a Avaliação

1. Certifique-se de estar na raiz do projeto (`project2/`) com o ambiente virtual ativo.
2. Verifique se as dependências do `requirements.txt` estão instaladas (incluindo `torch`, `scikit-learn`, `seaborn` e `tabulate`).
3. O checkpoint da rede neural deve estar posicionado em `data/processed/mlp_relation_pipeline_completo.pt`.
4. Rode o orquestrador pelo terminal:

```bash
python -m src.comparison.run_comparison