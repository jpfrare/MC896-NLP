# Resultados (gerado por src/report.py)

### Visão 1 — entidades

#### Tipagem de entidade dado o span (F1 por tipo)

| classe | n | P1 | LR |
|---|---|---|---|
| PATIENT | 11 | 0.78 | 0.96 |
| SYMPTOM | 16 | 0.69 | 0.62 |
| FINDING | 69 | 0.64 | 0.52 |
| DIAGNOSIS | 21 | 0.61 | 0.38 |
| PROCEDURE | 41 | 0.62 | 0.47 |
| VALUE | 32 | 0.44 | 0.79 |
| macro | 190 | 0.63 | 0.63 |
| acurácia | 190 | 0.62 | 0.56 |

P1 = dicionários de palavras-chave; LR = Regressão Logística com n-gramas 1–5. Mesmos spans do gold (sem NOT_ENTITY). Na linha 'acurácia', a coluna do modelo é a acurácia.

### Visão 2 — relação dado o par

#### F1 por classe, tipos do gold, só linhas positivas

| classe | n | TIPOS | P1_TABELA | LR | LR_TEXTO | MLP |
|---|---|---|---|---|---|---|
| SUPPORTS | 49 | 1.00 | 0.97 | 1.00 | 0.84 | 0.74 |
| HAS_VALUE | 30 | 1.00 | 1.00 | 1.00 | 0.48 | 0.75 |
| HAS_SYMPTOM | 29 | 1.00 | 0.51 | 1.00 | 0.81 | 0.78 |
| UNDERWENT_PROCEDURE | 14 | 1.00 | 0.00 | 1.00 | 0.44 | 0.00 |
| TREATED_BY | 3 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| macro | 125 | 1.00 | 0.50 | 1.00 | 0.51 | 0.45 |
| acurácia | 125 | 1.00 | 0.71 | 1.00 | 0.60 | 0.68 |

TIPOS = baseline que só olha (e1_type, e2_type). LR_TEXTO = a mesma LR sem os tipos (comparável ao MLP, que também só vê texto). Se TIPOS empata com a LR, a relação está sendo decidida pelos tipos.

#### Falsos alarmes nos pares NO_RELATION do gold

| modelo | n_negativos | falsos_alarmes | taxa | prediz_no_relation |
|---|---|---|---|---|
| TIPOS | 66 | 4 | 0.06 | True |
| P1_TABELA | 66 | 34 | 0.52 | True |
| LR | 66 | 8 | 0.12 | True |
| LR_TEXTO | 66 | 27 | 0.41 | True |
| MLP | 66 | 66 | 1.00 | False |

MLP não tem a classe NO_RELATION: toda predição dele é uma aresta.

#### Acurácia por caso (variação entre os 3 casos)

| modelo | PMC11131390_01 | PMC11722600_01 | PMC3859158_01 |
|---|---|---|---|
| TIPOS | 1.00 | 1.00 | 1.00 |
| P1_TABELA | 0.61 | 0.90 | 0.64 |
| LR | 1.00 | 1.00 | 1.00 |
| LR_TEXTO | 0.35 | 0.88 | 0.62 |
| MLP | 0.65 | 0.78 | 0.62 |

### Visão 3 — ponta a ponta

#### P1 e LR nos 3 casos do gold

| modelo | n_gold | n_pred | casadas | cobertura | acerto_rotulo | acerto_tipos | acerto_ponta_a_ponta | precisao_lim_inf | razao_pred_gold |
|---|---|---|---|---|---|---|---|---|---|
| P1 | 125 | 34 | 13 | 0.10 | 0.85 | 0.92 | 0.09 | 0.38 | 0.27 |
| LR | 125 | 91 | 21 | 0.17 | 0.52 | 0.38 | 0.09 | 0.23 | 0.73 |

cobertura = arestas do gold encontradas; acerto_rotulo/tipos = entre as encontradas; acerto_ponta_a_ponta = encontradas com o rótulo certo / arestas do gold; precisao_lim_inf é limite inferior se o gold não for exaustivo; razao_pred_gold > 1 indica excesso de arestas, < 1 indica perda.

#### Por caso

| modelo | caso | n_gold | n_pred | casadas | cobertura | acerto_rotulo | acerto_tipos | acerto_ponta_a_ponta | precisao_lim_inf | razao_pred_gold |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | PMC11131390_01 | 46 | 7 | 2 | 0.04 | 0.50 | 1.00 | 0.02 | 0.29 | 0.15 |
| P1 | PMC11722600_01 | 40 | 16 | 8 | 0.20 | 0.88 | 0.88 | 0.17 | 0.50 | 0.40 |
| P1 | PMC3859158_01 | 39 | 11 | 3 | 0.08 | 1.00 | 1.00 | 0.08 | 0.27 | 0.28 |
| LR | PMC11131390_01 | 46 | 39 | 7 | 0.15 | 0.43 | 0.14 | 0.07 | 0.18 | 0.85 |
| LR | PMC11722600_01 | 40 | 37 | 9 | 0.23 | 0.44 | 0.33 | 0.10 | 0.24 | 0.93 |
| LR | PMC3859158_01 | 39 | 15 | 5 | 0.13 | 0.80 | 0.80 | 0.10 | 0.33 | 0.38 |

#### Estrutura dos grafos (soma/média nos 3 casos)

| modelo | nos | arestas | grau_medio | componentes | pct_arestas_paciente | pct_negadas |
|---|---|---|---|---|---|---|
| GOLD | 150 | 125 | 1.64 | 29 | 0.34 | 0.23 |
| P1 | 56 | 34 | 1.25 | 22 | 0.42 | 0.28 |
| LR | 87 | 91 | 1.94 | 20 | 0.21 | 0.06 |

#### Arestas por tipo de relação

| relacao | GOLD | P1 | LR |
|---|---|---|---|
| SUPPORTS | 49 | 13 | 14 |
| HAS_VALUE | 30 | 9 | 22 |
| HAS_SYMPTOM | 29 | 5 | 2 |
| UNDERWENT_PROCEDURE | 14 | 0 | 10 |
| TREATED_BY | 3 | 7 | 43 |

### N-gramas

#### Ranking de bigramas: precisão@k contra o léxico de entidades anotadas

| criterio | 10 | 20 | 50 |
|---|---|---|---|
| n | 0.60 | 0.45 | 0.56 |
| npmi | 0.30 | 0.25 | 0.28 |
| pmi | 0.10 | 0.20 | 0.30 |

Limite inferior: só 15 dos 56 casos têm entidades anotadas.

#### Entidades compostas do gold que contêm uma colocação do corpus

| entidades_compostas_gold | com_colocacao | fracao | exemplos |
|---|---|---|---|
| 95 | 31 | 0.33 | physical examination, laboratory tests, complete blood count, CT scan, distal pancreatic mass, tumor cells, CT scan imaging, weight loss |

#### Pistas lexicais por relação (PMI n-grama × rótulo, 12 textos de treino)

| relacao | pista |
|---|---|
| HAS_SYMPTOM | `presented with`, `denied`, `diagnosed`, `loss`, `of weight`, `weight` |
| HAS_VALUE | `igg`, `were elevated`, `elevated`, `measuring`, `dose` |
| NO_RELATION | `g/dl`, `intraoperative`, `noted to`, `represented`, `noted`, `revealed that` |
| SUPPORTS | `revealed the`, `revealed a`, `demonstrated`, `showing`, `bilateral`, `of the left` |
| TREATED_BY | `closed`, `for presumed`, `presumed`, `infection`, `m`, `pneumoniae` |
| UNDERWENT_PROCEDURE | `received`, `and was given`, `clinic`, `first time`, `time and`, `visited` |

#### Arestas do gold que contêm uma pista do próprio rótulo

| relation_label | n | com_pista | fracao |
|---|---|---|---|
| HAS_SYMPTOM | 29 | 13 | 0.45 |
| HAS_VALUE | 30 | 4 | 0.13 |
| SUPPORTS | 49 | 6 | 0.12 |
| TREATED_BY | 3 | 0 | 0.00 |
| UNDERWENT_PROCEDURE | 14 | 1 | 0.07 |

#### Escolha do limiar de NPMI (12 textos de treino)

| limiar | recall_estrito_compostas_treino | candidatos |
|---|---|---|
| 0.200 | 0.613 | 1586 |
| 0.400 | 0.632 | 1631 |
| 0.600 | 0.516 | 1848 |
| 0.800 | 0.215 | 2091 |

#### Geradores de candidatos a entidade (recall nas entidades do gold)

| gerador | entidades | n | recall_estrito | recall_relaxado | candidatos |
|---|---|---|---|---|---|
| conteudo | 1 palavra | 52 | 0.846 | 0.962 | 332 |
| conteudo | compostas | 95 | 0.663 | 0.884 | 332 |
| pmi@0.2 | 1 palavra | 52 | 0.846 | 0.962 | 332 |
| pmi@0.2 | compostas | 95 | 0.663 | 0.884 | 332 |
| pmi | 1 palavra | 52 | 0.865 | 0.981 | 343 |
| pmi | compostas | 95 | 0.705 | 0.916 | 343 |
| pmi@0.6 | 1 palavra | 52 | 0.865 | 0.981 | 377 |
| pmi@0.6 | compostas | 95 | 0.663 | 0.947 | 377 |
| pmi@0.8 | 1 palavra | 52 | 0.942 | 1.000 | 448 |
| pmi@0.8 | compostas | 95 | 0.284 | 0.779 | 448 |
| noun_chunks | 1 palavra | 52 | 0.192 | 0.308 | 122 |
| noun_chunks | compostas | 95 | 0.432 | 0.537 | 122 |
| uniao | 1 palavra | 52 | 0.885 | 0.981 | 408 |
| uniao | compostas | 95 | 0.895 | 0.958 | 408 |

### LR por dentro

#### Fração do peso (|coef|) nas features de tipo (LR de relação)

| classe | massa_tipos | n_feats_tipos | n_feats_texto |
|---|---|---|---|
| HAS_SYMPTOM | 0.32 | 14 | 2000 |
| HAS_VALUE | 0.42 | 14 | 2000 |
| NO_RELATION | 0.15 | 14 | 2000 |
| SUPPORTS | 0.22 | 14 | 2000 |
| TREATED_BY | 0.32 | 14 | 2000 |
| UNDERWENT_PROCEDURE | 0.33 | 14 | 2000 |

#### Ablação do ngram_range

| ngram_range | f1_macro_entidades | n_entidades | f1_macro_relacoes | f1_macro_relacoes_so_texto | n_relacoes |
|---|---|---|---|---|---|
| 1-1 | 0.625 | 190 | 1.000 | 0.528 | 125 |
| 1-2 | 0.625 | 190 | 1.000 | 0.517 | 125 |
| 1-3 | 0.625 | 190 | 1.000 | 0.518 | 125 |
| 1-5 | 0.625 | 190 | 1.000 | 0.514 | 125 |

#### N-gramas que sobrevivem no vocabulário (min_df=2), por ordem

| ngram_range | classificador | 1-gramas | 2-gramas | 3-gramas | 4-gramas | 5-gramas |
|---|---|---|---|---|---|---|
| 1-1 | entidade | 299 | 0 | 0 | 0 | 0 |
| 1-1 | relacao | 489 | 0 | 0 | 0 | 0 |
| 1-1 | relacao_so_texto | 489 | 0 | 0 | 0 | 0 |
| 1-2 | entidade | 299 | 54 | 0 | 0 | 0 |
| 1-2 | relacao | 489 | 751 | 0 | 0 | 0 |
| 1-2 | relacao_so_texto | 489 | 751 | 0 | 0 | 0 |
| 1-3 | entidade | 299 | 54 | 11 | 0 | 0 |
| 1-3 | relacao | 489 | 751 | 642 | 0 | 0 |
| 1-3 | relacao_so_texto | 489 | 751 | 642 | 0 | 0 |
| 1-5 | entidade | 299 | 54 | 11 | 2 | 0 |
| 1-5 | relacao | 418 | 539 | 430 | 343 | 270 |
| 1-5 | relacao_so_texto | 418 | 539 | 430 | 343 | 270 |

Explica curvas planas na ablação: n-gramas longos raramente aparecem 2 vezes no treino.

#### N-gramas com maior coeficiente por classe

| tarefa | classe | ngrama |
|---|---|---|
| entidade | DIAGNOSIS | `aneurysm`, `ptx`, `vsd`, `obstruction`, `pulmonary`, `thrombosis` |
| entidade | FINDING | `normal`, `lesion`, `level`, `mass`, `rate`, `formation` |
| entidade | NOT_ENTITY | `be`, `present`, `reveal`, `demonstrate`, `discharge`, `limb` |
| entidade | PATIENT | `patient`, `old`, `she`, `he`, `year old`, `boy` |
| entidade | PROCEDURE | `test`, `ct`, `oral`, `therapy`, `examination`, `tube` |
| entidade | SYMPTOM | `fever`, `pain`, `rash`, `abdominal`, `distress`, `fatigue` |
| entidade | VALUE | `mg`, `dl`, `mm`, `cm`, `ml`, `week` |
| relacao | HAS_SYMPTOM | `have`, `develop`, `present`, `present with`, `not`, `in` |
| relacao | HAS_VALUE | `of`, `be`, `dose`, `elevated`, `be elevated`, `on the` |
| relacao | NO_RELATION | `and`, `with no`, `with`, `no`, `he`, `use` |
| relacao | SUPPORTS | `reveal`, `show`, `demonstrate`, `consistent with`, `consistent`, `find` |
| relacao | TREATED_BY | `be`, `require`, `improve`, `obstruction`, `secondary to`, `secondary` |
| relacao | UNDERWENT_PROCEDURE | `undergo`, `for`, `receive`, `on`, `be start`, `start` |
