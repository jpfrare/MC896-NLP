"""Tipagem de entidades do Projeto 1 (dicionários de palavras-chave), aplicada a spans dados.

As listas NÃO são copiadas: são lidas do próprio notebook do P1
(pipelines/pipeline1/projeto1_modificado.ipynb) com `ast`, de modo que qualquer
alteração nas regras do P1 se reflete automaticamente na comparação.

Fidelidade às regras originais (célula 16 do notebook):
  - PATIENT  se o span contém um termo de paciente (patient_terms)
  - VALUE    se o span casa com a regex de medidas do P1 (célula 7)
  - senão, testa por substring na ordem SYMPTOM -> DIAGNOSIS -> FINDING -> PROCEDURE,
    com FINDING como padrão (igual ao P1, inclusive os falsos positivos de substring,
    p.ex. "ct" dentro de outras palavras).
Diferença conhecida: o P1 testava sobre o rótulo lematizado; aqui usamos o texto em
minúsculas, sem dígitos e pontuação (o teste por substring torna a diferença pequena).
"""
import ast
import json
import re
from functools import lru_cache

from .common import PIPELINE1

NOTEBOOK = PIPELINE1 / "projeto1_modificado.ipynb"
LIST_NAMES = ["symptom_keywords", "finding_keywords", "diagnosis_keywords", "exam_keywords", "patient_terms"]
# Regex de medidas da célula 7 do notebook do P1 (é uma chamada re.compile, não um literal).
MEASUREMENT_RE = re.compile(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(cm|mm|ng/ml|iu/ml|mg)", re.IGNORECASE)


@lru_cache(maxsize=1)
def p1_lists() -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in LIST_NAMES:
                    found[name] = list(ast.literal_eval(node.value))
    missing = set(LIST_NAMES) - set(found)
    if missing:
        raise RuntimeError(f"Listas do P1 não encontradas em {NOTEBOOK.name}: {sorted(missing)}")
    return found


def clean_label(text: str) -> str:
    toks = re.findall(r"[a-z]+(?:[-/][a-z]+)*", str(text).lower())
    return " ".join(toks)


def p1_type(text: str) -> str:
    lists = p1_lists()
    label = clean_label(text)
    tokens = set(label.split())
    if tokens & set(lists["patient_terms"]):
        return "PATIENT"
    if MEASUREMENT_RE.search(str(text)):
        return "VALUE"
    if any(w in label for w in lists["symptom_keywords"]):
        return "SYMPTOM"
    if any(w in label for w in lists["diagnosis_keywords"]):
        return "DIAGNOSIS"
    if any(w in label for w in lists["finding_keywords"]):
        return "FINDING"
    if any(w in label for w in lists["exam_keywords"]):
        return "PROCEDURE"
    return "FINDING"


if __name__ == "__main__":
    print({k: len(v) for k, v in p1_lists().items()})
    for t in ["44-year-old man", "vague epigastric pain", "CT scan", "pancreatitis", "6 cm", "cyst"]:
        print(f"{t!r:28} -> {p1_type(t)}")
