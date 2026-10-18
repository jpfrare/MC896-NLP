# src/ — camada comum do Projeto 2

Código compartilhado para que todos os modelos sejam medidos do mesmo jeito.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Arquivos

| Arquivo | O que faz |
|---|---|
| `gold.py` | Junta `data/annotations/text*.csv` em `data/processed/gold.csv` (esquema único) e valida os rótulos |
| `evaluation.py` | P/R/F1 por classe, macro-F1, matriz de confusão e validação cruzada deixando um texto de fora |
| `baselines.py` | Baselines (classe majoritária, só tipos, TF-IDF + LR) e confronto com as regras do Projeto 1 |

## Rodar

A partir de `project2/`:

```bash
python src/gold.py        # regenera data/processed/gold.csv
python src/baselines.py   # imprime tudo e salva data/processed/baseline_results.csv
```

## Como plugar o seu modelo

Esquema do gold: `doc_id, e1_text, e1_type, e2_text, e2_type, context, relation`.
Seu modelo vira uma função que treina em `train` e devolve uma lista de rótulos para `test`:

```python
import sys; sys.path.append("../../src")      # em notebook dentro de pipelines/notebooks
from gold import load_gold
from evaluation import leave_one_doc_out, print_report, summary_row

def fit_predict(train, test):
    ...                       # treine com train, preveja test
    return preds              # lista com len(test) rotulos de gold.LABELS

res = leave_one_doc_out(load_gold(), fit_predict)
print_report(res, "meu_modelo")
print(summary_row(res, "meu_modelo"))   # linha para a tabela do README
```

## Treino e teste (combinado do grupo)

- `data/annotations/`      anotacoes de treino (Frare/Setti). Vira `data/processed/gold.csv`.
- `data/annotations_test/` textos anotados SO para teste, nunca usados no treino. Vira `gold_test.csv`.
  Mesmo formato dos `text*.csv`, mesmos tipos de no (`NODE_TYPES`) e mesmas relacoes (`LABELS`).

```python
from gold import load_gold
from evaluation import train_test, compare_table

train, test = load_gold("train"), load_gold("test")
results = {"regressao_logistica": train_test(train, test, fit_predict_lr),
           "rede_neural": train_test(train, test, fit_predict_nn)}
print(compare_table(results))      # accuracy, macro-F1 e F1 por classe, uma linha por modelo
```

`train_test` e `load_gold("test")` levantam erro se algum texto aparecer no treino e no teste.
`python src/baselines.py` roda tudo e salva `data/processed/comparison_test.csv` quando o teste existe.

## Etapa 1 (tipo do no) e etapa 2 (relacao)

O gold tem duas tarefas, e o modelo de dois estagios do Frare/Setti cobre as duas:

- etapa 1: texto da entidade -> tipo (`build_entities(gold)`, coluna `type`, `labels=NODE_TYPES`)
- etapa 2: par de entidades + trecho entre elas -> relacao (`load_gold()`, coluna `relation`)

```python
from gold import load_gold, build_entities, NODE_TYPES
from evaluation import leave_one_doc_out

ents = build_entities(load_gold())
res = leave_one_doc_out(ents, fit_predict_nos, labels=NODE_TYPES, target="type")
```

## Cuidados

- Sempre compare com `types_only`. Nas anotações atuais cada par de tipos tem um único
  rótulo, então um modelo que recebe os tipos acerta quase tudo sem olhar o texto.
  O teste que mede n-gramas é o que usa só `context`.
- `evaluate` levanta erro se houver rotulo vazio ou fora de `LABELS`.
- O gold e pequeno (4 textos). Reporte a media e o desvio entre textos, nao so o numero geral.
- Nas regras do P1, "NO_RELATION" significa "as regras nao extrairam esse par".
  As arestas das regras sem par no gold nao podem ser julgadas (o gold nao e exaustivo).