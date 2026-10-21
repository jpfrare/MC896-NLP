# `src/` — sistema de comparação, análises de n-gramas e painel

## Como rodar (um comando)

A partir de `project2/`:

```fish
python3 run.py
```

Funciona com qualquer Python 3.8+ e faz quatro coisas:

1. cria o ambiente `project2/.venv` com **Python 3.12** (via [uv](https://docs.astral.sh/uv/), que baixa o 3.12 sozinho; sem uv, usa um `python3.12` do sistema);
2. instala o `requirements.txt` (só na primeira vez ou quando o arquivo muda);
3. roda todas as etapas (`python -m src.run_all`), atualizando as tabelas do README;
4. abre o painel `assets/html/index.html` no navegador.

| Opção | Efeito |
|---|---|
| `--rapido` | pula a ablação de n-gramas (a etapa mais lenta) |
| `--reinstalar` | apaga e recria o `.venv` |
| `--nao-abrir` | não abre o navegador |

Para instalar o uv: Arch `sudo pacman -S uv`; macOS `brew install uv`; outros sistemas, veja a documentação do uv. A primeira execução precisa de internet, para baixar os pacotes e os modelos spaCy.

**Por que Python 3.12:** o `en_core_sci_sm` 0.5.4 foi treinado com spaCy 3.7.4 e só carrega com spaCy 3.7. O spaCy 3.7 só tem pacotes até o Python 3.12 e foi compilado contra NumPy 1.x. O `src/check_env.py`, primeira etapa do `run_all`, confere tudo isso e testa os dois modelos spaCy.

**No VS Code:** para os notebooks, selecione o interpretador `project2/.venv/bin/python`.

## O que é gerado

| Saída | Onde | Commitar? |
|---|---|---|
| Previsões no esquema único | `data/processed/comparison/pred_*.csv`, `ent_*.csv` | sim |
| Tabelas de métricas | `data/processed/comparison/tables/*.csv` | sim |
| Figuras | `assets/images/*.png` | sim (o README usa) |
| Painel interativo | `assets/html/index.html` | sim |
| Relatório em Markdown | `assets/resultados.md` e o trecho entre marcadores do `README.md` | sim |
| Ambiente e cache | `.venv/`, `__pycache__/` | não (`.gitignore`) |

O `run_all` apaga as saídas anteriores antes de começar, então nenhuma tabela mistura execuções. Se alguma etapa falhar, o resumo final, o README e o painel mostram um aviso de "resultados incompletos".

## Etapas avulsas

Com o ambiente ativo (`source .venv/bin/activate.fish`), a partir de `project2/`:

```fish
python -m src.check_env                 # ambiente + teste rápido dos modelos spaCy
python -m src.comparison.adapt_p1       # P1: e2e, tabela tipo->rótulo, tipagem de spans
python -m src.comparison.baselines      # baseline TIPOS
python -m src.comparison.run_lr         # LR e LR_TEXTO (par), entidades e e2e
python -m src.comparison.infer_mlp      # MLP (--best para os pesos de menor loss)
python -m src.comparison.check          # sanidade: TIPOS deve dar acurácia 1.000 na prévia
python -m src.comparison.evaluate       # métricas e figuras das 3 visões
python -m src.ngrams.collocations       # PMI/NPMI no corpus
python -m src.ngrams.relation_cues      # pistas lexicais por relação
python -m src.ngrams.candidates         # NPMI como gerador de candidatos a entidade
python -m src.ngrams.lr_features        # coeficientes da LR e ablação do ngram_range
python -m src.graph.logical_model       # figura do modelo lógico
python -m src.report                    # tabelas no README e em assets/resultados.md
python -m src.graph.dashboard           # painel assets/html/index.html
```

## O painel (`assets/html/index.html`)

É um arquivo único que abre direto no navegador. A biblioteca de grafos vem de CDN, então ele precisa de internet. Tem três abas:

- **Grafos por caso:** escolha o caso e o modelo (P1 ou LR; "GOLD" mostra só a anotação). As arestas aparecem coloridas por status: certa, rótulo errado, a mais (não está no gold) e perdida (só no gold, tracejada). Na mesma aba há:
  - filtros por status e por relação;
  - opção de ocultar o nó PATIENT (o hub que polui o grafo) e de mostrar os rótulos das arestas;
  - busca de entidade;
  - detalhes ao clicar num nó ou aresta, com o contexto usado pelo modelo e o trecho do texto clínico destacado;
  - tabela das arestas visíveis (clicar numa linha foca a aresta no grafo).
- **Grafo do corpus (P1):** as arestas do P1 nos 55 casos, com as entidades unificadas pelo texto normalizado. O tamanho do nó mostra o grau, há filtro por grau mínimo e por relação.
- **Métricas:** as mesmas figuras e tabelas do README.

Nós e arestas preditos são alinhados ao gold pelo mesmo critério da avaliação (`comparison/match.py`).

## Estrutura

```
project2/
├── run.py                  cria o ambiente, instala, roda tudo e abre o painel
└── src/
    ├── run_all.py          orquestra as etapas (limpa as saídas antes)
    ├── check_env.py        checa Python/NumPy/spaCy e testa os modelos spaCy
    ├── report.py           seções de resultados -> README.md, assets/resultados.md e painel
    ├── comparison/
    │   ├── common.py       caminhos, esquemas, cores fixas por modelo, utilidades
    │   ├── gold.py         gold (linhas, entidades, definição única de entidade composta) e treino
    │   ├── adapt_p1.py     adaptador do P1
    │   ├── p1_typing.py    tipagem por palavras-chave do P1 (listas lidas do notebook)
    │   ├── baselines.py    baseline TIPOS
    │   ├── run_lr.py       LR / LR_TEXTO (usa useful/train.py; e2e lido do notebook)
    │   ├── infer_mlp.py    inferência do MLP a partir do checkpoint
    │   ├── match.py        casamento de arestas com o gold (visão ponta a ponta)
    │   ├── evaluate.py     métricas das 3 visões + figuras
    │   ├── plots.py        estilo dos gráficos
    │   └── check.py        sanidade do esquema
    ├── graph/
    │   ├── dashboard.py    painel interativo
    │   └── logical_model.py figura do modelo lógico
    └── ngrams/
        ├── text.py         tokenização, contagens e PMI/NPMI
        ├── collocations.py colocações e precisão@k contra o léxico anotado
        ├── relation_cues.py n-gramas associados a cada relação
        ├── candidates.py   segmentação por NPMI como gerador de candidatos
        └── lr_features.py  coef_ da LR, peso tipos × texto, ablação e vocabulário por ordem
```

Nenhum arquivo dos pipelines originais é modificado. O `src/` só lê os CSVs, o checkpoint do MLP, as classes de `pipelines/pipeline2/useful/` e, via `ast`, as listas de palavras-chave do notebook do P1 e a função `process_cases_to_graph_csvs` do notebook do LR.

## Esquemas de saída

**Arestas**, em `pred_<modelo>_<visão>.csv`:

| coluna | conteúdo |
|---|---|
| `model` | `P1`, `P1_TABELA`, `LR`, `LR_TEXTO`, `MLP`, `MLP_BEST`, `TIPOS` |
| `view` | `par` (par e tipos do gold; prediz só a relação) ou `e2e` (extrai do texto) |
| `case_id` | caso do MultiCaRe |
| `gold_row` | id da linha do gold (só em `par`) |
| `e1_text`, `e1_type`, `e2_text`, `e2_type` | entidades |
| `relation_label` | rótulo predito (pode ser `NO_RELATION`) |
| `intermediate_text` | contexto usado pelo modelo |
| `negated` | marcador de negação no contexto (descritivo; lista de negação do P1) |

**Entidades**, em `ent_<modelo>.csv`: `model, case_id, ent_id, text, gold_type, pred_type`.

**Cores fixas por modelo** em todos os gráficos (Okabe-Ito, validada para daltonismo):

| Modelo | Cor | Codificação extra |
|---|---|---|
| P1 / P1_TABELA | `#D55E00` | P1_TABELA hachurado |
| LR / LR_TEXTO | `#0072B2` | LR_TEXTO hachurado |
| MLP | `#009E73` | — |
| TIPOS | `#999999` | hachurado |
| MLP_BEST | `#CC79A7` | hachurado |

## Reprodutibilidade

- Seeds fixas: LR com `random_state=42`. A inferência do MLP é determinística (`eval()`), e o `run.py` fixa `PYTHONHASHSEED=0`.
- Empates nos rankings de n-gramas são desfeitos em ordem alfabética, sem depender do hash de strings.
- Os arquivos de anotação são lidos em ordem alfabética. O notebook original usa a ordem do sistema de arquivos, então o LR pode diferir marginalmente do notebook.
- O limiar de NPMI do gerador de candidatos é escolhido nos 12 textos de treino, nunca no gold.
- "Entidade clínica" e "entidade composta" têm uma única definição (`gold.load_clinical_entities`).
