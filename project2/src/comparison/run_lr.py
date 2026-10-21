"""Adaptador da Regressão Logística (pipeline2).

  pred_LR_par.csv        RelationClassifier de useful/train.py (n-gramas + tipos), treinado nos
                         12 textos anotados com os hiperparâmetros do notebook; classifica os pares do gold.
  pred_LR_TEXTO_par.csv  o mesmo classificador SEM a informação de tipo (tipos trocados por uma
                         constante): mede o que os n-gramas do texto sozinhos recuperam e é o
                         comparável direto do MLP, que também só vê texto.
  ent_LR.csv             EntityClassifier (n-gramas 1..5) tipando os spans do gold.
  pred_LR_e2e.csv        extração ponta a ponta nos 3 casos do gold, com a função
                         `process_cases_to_graph_csvs` lida do próprio notebook
                         pipelines/pipeline2/pipeline_ml_tradicional.ipynb (não é copiada nem editada).

Nada em useful/ nem no notebook é alterado. As funções train_* são reutilizadas por src/ngrams.
"""
import ast
import json
import sys
from pathlib import Path

import pandas as pd

from .common import (GOLD_IDS, OUT, PIPELINE2, RAW, save_entity_predictions, save_predictions,
                     to_schema)
from .gold import load_gold, load_gold_entities, load_training_annotation_list, load_training_annotations

sys.path.insert(0, str(PIPELINE2))
from useful.preprocess import MedicalTextProcessor  # noqa: E402
from useful.train import CSVReader, EntityClassifier, RelationClassifier  # noqa: E402

NOTEBOOK = PIPELINE2 / "pipeline_ml_tradicional.ipynb"
E2E_FUNCTION = "process_cases_to_graph_csvs"
NO_TYPE = "ANY"  # tipo constante usado na variante só-texto


def train_relation(text_only: bool = False, **kwargs) -> RelationClassifier:
    train = load_training_annotations()
    e1 = pd.Series(NO_TYPE, index=train.index) if text_only else train["e1_type"]
    e2 = pd.Series(NO_TYPE, index=train.index) if text_only else train["e2_type"]
    clf = RelationClassifier(C=0.5, **kwargs)  # igual à célula 4 do notebook
    clf.fit(e1, e2, train["intermediate_text"], train["relation_label"])
    return clf


def train_entity(**kwargs) -> EntityClassifier:
    reader = CSVReader(load_training_annotation_list())  # igual às células 2 e 4 do notebook
    clf = EntityClassifier(C=0.5, **kwargs)
    clf.fit(reader.entities["data"], reader.entities["label"])
    return clf


def notebook_function(name: str = E2E_FUNCTION):
    """Lê a definição da função no notebook (fonte única) e a executa num namespace isolado."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                ns = {"pd": pd, "Path": Path, "MedicalTextProcessor": MedicalTextProcessor,
                      "EntityClassifier": EntityClassifier, "RelationClassifier": RelationClassifier}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), ns)
                return ns[name]
    raise RuntimeError(f"Função {name} não encontrada em {NOTEBOOK.name}")


def run_pair(clf: RelationClassifier, text_only: bool = False):
    pair = load_gold()
    e1 = pd.Series(NO_TYPE, index=pair.index) if text_only else pair["e1_type"]
    e2 = pd.Series(NO_TYPE, index=pair.index) if text_only else pair["e2_type"]
    pair["relation_label"] = clf.predict(e1, e2, pair["intermediate_text"]).values
    name = "LR_TEXTO" if text_only else "LR"
    save_predictions(to_schema(pair, name, "par"), name, "par")


def run_entities(clf: EntityClassifier):
    ents = load_gold_entities()
    ents["pred_type"] = clf.predict(ents["text"].astype(str)).values
    save_entity_predictions(ents, "LR")


def run_e2e(ent_clf: EntityClassifier, rel_clf: RelationClassifier):
    cases = pd.read_csv(RAW / "cases.csv")
    cases = cases[cases["case_id"].isin(GOLD_IDS)]
    OUT.mkdir(parents=True, exist_ok=True)
    process = notebook_function()
    graph = process(cases_df=cases, entity_classifier=ent_clf, relation_classifier=rel_clf,
                    output_graph_path=OUT / "lr_graph_gold_bruto.csv")
    missing = set(GOLD_IDS) - set(graph["case_id"])
    if missing:
        print(f"[LR/e2e] aviso: casos do gold sem nenhuma aresta: {sorted(missing)}")
    save_predictions(to_schema(graph, "LR", "e2e"), "LR", "e2e")


def main():
    ent_clf = train_entity()
    rel_clf = train_relation()
    run_pair(rel_clf)
    run_pair(train_relation(text_only=True), text_only=True)
    run_entities(ent_clf)
    run_e2e(ent_clf, rel_clf)


if __name__ == "__main__":
    main()
