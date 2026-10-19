# Relatório de Comparação de Modelos

A avaliação considera as 5 classes de relações válidas, omitindo NO_RELATION e NOT_ENTITY.

**Distribuição do Gold Analisado:** 125 instâncias.

## 1. Métricas de Desempenho (Relações)
| Modelo               |   Macro-F1 |   Precision |    Recall |
|:---------------------|-----------:|------------:|----------:|
| Projeto 1 (Regras)   |   0.132949 |    0.6      | 0.0756979 |
| LR N-gramas (Manual) |   0.650614 |    0.662065 | 0.645236  |
| MLP (PyTorch)        |   0.453853 |    0.442074 | 0.473263  |

## 2. Estatísticas do Grafo (Projeto 1)
- **Total de Nós**: 5867
- **Total de Arestas**: 654
- **Tipos de Nós Únicos**: 6
- **Tipos de Relações Únicas**: 0
