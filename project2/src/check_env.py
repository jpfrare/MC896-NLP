"""Checa se o ambiente é o esperado pelo projeto e faz um teste rápido dos modelos spaCy.

Por que o ambiente é fixo:
  o en_core_sci_sm 0.5.4 foi treinado com spaCy 3.7.4 e só carrega com spaCy 3.7.x;
  o spaCy 3.7 só tem pacotes até o Python 3.12 e foi compilado contra NumPy 1.x.
  Logo: Python 3.12 + spaCy 3.7 + NumPy < 2 (ver requirements.txt).

Uso (a partir de project2/):  python -m src.check_env
É a primeira etapa de `python -m src.run_all`.
"""
import importlib.util
import json
import sys
from pathlib import Path

SAMPLE = "The patient underwent computed tomography, which revealed a cystic lesion in the pancreas."


def require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg + " Recrie o ambiente com `python3 run.py --reinstalar` (ver src/README.md).")


def main():
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  Python {sys.version.split()[0]} ({sys.executable}; ambiente virtual {'ativo' if in_venv else 'NÃO ativo'})")
    require(sys.version_info[:2] == (3, 12), "O projeto usa Python 3.12.")

    import numpy
    print(f"  NumPy {numpy.__version__}")
    require(int(numpy.__version__.split(".")[0]) < 2, "O spaCy 3.7 exige NumPy < 2.")

    import spacy
    from spacy.util import is_compatible_version
    spec = importlib.util.find_spec("en_core_sci_sm")
    require(spec is not None, "en_core_sci_sm não está instalado.")
    meta_path = next(Path(spec.origin).parent.glob("**/meta.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"  spaCy {spacy.__version__} · en_core_sci_sm {meta.get('version')} (exige spaCy {meta.get('spacy_version')})")
    require(bool(is_compatible_version(spacy.__version__, meta.get("spacy_version", ""))),
            "Versão do spaCy incompatível com o en_core_sci_sm.")

    for model in ("en_core_sci_sm", "en_core_web_sm"):
        nlp = spacy.load(model, disable=["ner"])
        doc = nlp(SAMPLE)
        print(f"  {model}: componentes={nlp.pipe_names} vetores={nlp.vocab.vectors.shape}")
        print(f"    lemas: {[t.lemma_ for t in doc if not t.is_punct]}")
        print(f"    noun chunks: {[c.text for c in doc.noun_chunks]}")

    import torch
    import sklearn
    print(f"  torch {torch.__version__} · scikit-learn {sklearn.__version__}")


if __name__ == "__main__":
    main()
