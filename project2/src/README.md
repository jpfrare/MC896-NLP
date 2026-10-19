# src — Sistema de comparação (Projeto 1 × Projeto 2)

Código que **não roda dentro de notebook**: compara as regras do Projeto 1 com os modelos de
N-gramas + ML do Projeto 2 e gera tabelas, gráficos e um relatório em Markdown.

## Instalação (Linux, a partir da raiz do repositório)

```bash
cd project2
python3 -m venv .venv                      
source .venv/bin/activate                  
pip install torch --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU (leve)
pip install -r requirements.txt
```

Opcional (lematizador científico usado pelos colegas; sem ele o código usa `en_core_web_sm` e,
se faltar, uma tokenização simples — o backend usado aparece no relatório):

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

## Execução

```bash
cd project2
python -m src.comparison.run_comparison              # tudo (treina LR, MLP, NB, SVM, RF)
python -m src.comparison.run_comparison --no-mlp     # sem PyTorch
python -m src.comparison.run_comparison --no-extras  # só LR/MLP + Projeto 1
pytest tests -q                                      # teste de fumaça
```

Saídas: `data/processed/comparison/` (CSVs, predições uniformizadas, `comparison_report.md`) e
`assets/images/comparison/` (8 gráficos PNG).

## Módulos (`src/comparison/`)

| arquivo | função |
|---|---|
| `config.py`, `data.py` | caminhos e carregamento uniforme (treino manual, gold, weak supervision, saída do P1) |
| `p1_baseline.py` | Projeto 1 lido da saída oficial (`nodes.csv`/`edges.csv`) com casamento relaxado de menções |
| `models.py` | treina/aplica LR (código em `Train.py`), NB, SVM, RF e a ablação de n-gramas |
| `mlp_torch.py` | MLP PyTorch (mesma arquitetura do notebook `pipeline_rede_neural`), treino determinístico |
| `metrics.py` | P/R/F1, matriz de confusão, IC 95% por bootstrap |
| `graph_analysis.py` | densidade, grau, arestas em excesso/perdidas, esquema tipo→relação→tipo |
| `plots.py`, `run_comparison.py` | gráficos e orquestração/relatório |

## Decisões de projeto

- **Código**: somente `pipelines/notebooks/Train.py` foi alterado (import do spaCy opcional e
  lematizador com *fallback*). Nenhum notebook foi modificado.
- **Gold**: `data/annotations/gold01..03.csv` (3 documentos, 191 pares) — fora do treino manual e da weak supervision.
- **O `.pth` do MLP não é reutilizado**: o `TfidfVectorizer` e o mapa de rótulos não foram salvos; por isso o MLP é retreinado (seed fixa).
