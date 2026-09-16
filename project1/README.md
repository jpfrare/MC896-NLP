# Projeto Extração de Grafo de Conhecimento Clínico do MultiCaRe

# Project MultiCaRe Clinical Knowledge Graph Extraction

## Slides

[Extração de Grafos de Casos Clínicos (MultiCaRe) - NoLarPing.pdf](https://github.com/jpfrare/MC896-NLP/blob/c4b3e05bec85a9cb38199fd3f559e87298f27c0b/project1/assets/Extra%C3%A7%C3%A3o%20de%20Grafos%20de%20Casos%20Cl%C3%ADnicos%20(MultiCaRe)%20-%20NoLarPing.pdf)

## Metodologia

O projeto realiza a extração de dados de casos clínicos para a construção de um grafo de conhecimento. A arquitetura de extração foi baseada no uso de técnicas clássicas de Processamento de Linguagem Natural e dividida nas seguintes etapas:

* **Ingestão e Preparação:** Leitura dos arquivos `cases.csv` e `metadata.csv` e mesclagem dos dados.

* **Extração de Medidas (RegEx):** Uso de expressões regulares para buscar valores numéricos vinculados a unidades de medida (como `cm`, `mm`, `ng/ml`, `iu/ml`).
    ```python
    measurement_pattern = re.compile(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*[xX]\s*\d+(?:\.\d+)?)*|\d+(?:\.\d+)?(?:\s*[xX]\s*\d+(?:\.\d+)?)*)\s*(cm|mm|ng/ml|iu/ml|mg)')
    ```

* **Extração de Entidades:** Utilização do `en_core_web_sm` da biblioteca spaCy. A extração de pacientes foi feita por meio do `Matcher` utilizando vocabulários fechados (ex: "woman", "man", "patient"). Para conceitos clínicos (doenças, exames, tratamentos), utilizamos análise de classe gramatical (adjetivos seguidos de substantivos) em conjunto com remoção de *stopwords* customizadas e via NLTK. As entidades foram classificadas em categorias como `DISEASE` e `Procedure/Exam` através de verificação em listas de palavras-chave (dicionários estáticos).

* **Tratamento de Negação:** Implementação de uma heurística de janela de contexto analisando os tokens anteriores às entidades. Se palavras associadas a negação (como "no", "deny" ou "without") antecedem o termo, a extração é ignorada, prevenindo a geração de falsos positivos (ex: "no nausea").

* **Identificação de Relações (Arestas):** As arestas sintáticas foram estabelecidas percorrendo os limites de sentenças (via `doc.sents` do spaCy). A relação foi inferida baseada nos lemas dos verbos presentes na sentença (ex: "present" $\rightarrow$ `HAS_SYMPTOM`; "undergo" $\rightarrow$ `UNDERWENT_PROCEDURE`). 

* **Heurística Espacial com Viés Direcional:** Para conectar valores numéricos aos seus respectivos conceitos, é calculado espaço livre em caracteres entre as duas entidades. Existe uma penalidade direcional, de modo queo algoritmo aceita um espaço maior (30 caracteres) se o nome do exame preceder o valor, mas exige adjacência quase estrita se o valor preceder a entidade (ex: "6 cm cyst"), resolvendo ambiguidades de fronteira de oração.

## Trabalhos Estudados

* **An Open-Source Clinical Case Dataset... (Nievas Offidani et al., 2025):** Artigo que originou a base de dados MultiCaRe (focada em relatos extraídos do PubMed), de onde retiramos os textos de base.

* **A Survey on Knowledge Graphs... (Ji et al., 2022):** Estudo de literatura recomendado como base teórica para entender a representação, aquisição e unificação dos grafos.

## Modelo Lógico

O modelo desenhado baseia-se em um esquema de grafos de propriedades com tipagem de nós e relacionamentos fixos.

* **Nós Extraídos:** `Patient`, `DISEASE`, `Procedure/Exam`, `MedicalConcept`, `ExamResult`.

* **Arestas/Relacionamentos:** `HAS_SYMPTOM`, `UNDERWENT_PROCEDURE`, `SUPPORTS`, `TREATED_BY` e `HAS_VALUE`.

[Lembre-se de substituir o arquivo abaixo pelo PNG do diagrama lógico gerado pelo grupo na pasta correspondente]

## Análises que podem ser realizadas

A partir da estrutura extraída nas tabelas finais de nós e arestas, diversas análises podem ser conduzidas:

* **Trajetória de Diagnóstico:** Identificação da jornada clínica a partir de `HAS_SYMPTOM`, passando pelos exames realizados (`UNDERWENT_PROCEDURE`) e os tratamentos executados (`TREATED_BY`).
* **Associação Quantitativa:** Verificação dos valores extraídos (`HAS_VALUE`) conectados a resultados específicos, permitindo buscar anomalias ou padrões laboratoriais associados a doenças específicas (`DISEASE`).
* **Correlações:** Usando as arestas geradas pelo verbo "reveal" (`SUPPORTS`), podemos analisar quais tipos de exames de imagem ou procedimentos suportam mais rapidamente um conceito médico.

## Ferramentas

* **Python, Pandas & Regex:** Linguagem base e ferramentas usadas para estruturação tabular, merge dos datasets brutos e captura de métricas formatadas em string.

* **spaCy:** A biblioteca principal do projeto. Usada em toda a infraestrutura NLP para realizar tokenização, lematização, mapeamento de classes gramaticais (POS tagging), delineamento de sentenças e varreduras com o módulo `Matcher`.

* **NLTK:** Utilizada de forma auxiliar para download e filtragem do corpus *stopwords* de inglês.

* **PyVis:** Escolhida para prover uma interface visual interativa de grafos de rede no próprio notebook e no navegador via HTML, possibilitando cores dinâmicas, física simulada (nodes *Force Atlas*) e realce com clique.

## Resultados

A estratégia mista baseada em spaCy e dicionários se demonstrou funcional para os casos de validação. O projeto unifica os resultados no esquema tabular de nós e arestas sugerido pela disciplina.

Como destaque positivo de apresentação, implementou-se uma visualização interativa do grafo final. Os nós gerados foram colorizados segundo as heurísticas extraídas: Verde para o paciente (`Patient`), Vermelho para doenças (`DISEASE`), Azul para conceitos relevantes fora de escopo (`MedicalConcept`), Amarelo para exames e procedimentos (`Procedure/Exam`) e Cinza para valores extraídos do texto (`ExamResult`).

<img src="https://github.com/jpfrare/MC896-NLP/blob/main/project1/assets/knowledge_graph.png" width="75%">

## Como Modelos de Linguagem foram Usados

* Auxilio com dúvidas referentes a sintaxe e configuração de bibliotecas

* Auxilio na fabricação dos slides utilizados na apresentação

* Extruturação do código HTML para representação visual do grafo de conhecimento

## Referências Bibliográficas

* Nievas Offidani, M., Roffet, F., González Galtier, M. C., Massiris, M., & Delrieux, C. (2025). An Open-Source Clinical Case Dataset for Medical Image Classification and Multimodal AI Applications. Data 2025, 10(8), 123. [https://doi.org/10.3390/DATA10080123](https://www.google.com/search?q=https://doi.org/10.3390/DATA10080123)
* Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2022). A Survey on Knowledge Graphs: Representation, Acquisition, and Applications. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494–514. [https://doi.org/10.1109/TNNLS.2021.3070843](https://www.google.com/search?q=https://doi.org/10.1109/TNNLS.2021.3070843)
