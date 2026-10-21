"""Tokenização e contagem de n-gramas compartilhadas pelas análises de n-gramas.

Deliberadamente simples (regex + stopwords do scikit-learn), sem spaCy: assim a
análise de n-gramas roda em qualquer ambiente e não depende de nenhum modelo.
"""
import math
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*|[^\sa-z0-9]")
STOP = frozenset(ENGLISH_STOP_WORDS) | {"patient", "year", "old"}  # 'patient' é PATIENT, tratado à parte
SENT_RE = re.compile(r"(?<=[.!?;])\s+")


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(str(text).lower())


def is_content(tok: str) -> bool:
    return tok[0].isalpha() and tok not in STOP and len(tok) > 1


def sentences(text: str) -> list:
    return [tokenize(s) for s in SENT_RE.split(str(text)) if s.strip()]


class NgramStats:
    """Contagens de 1..3-gramas dentro de sentenças e PMI (Church & Hanks, 1990)."""

    def __init__(self, texts):
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        for text in texts:
            for toks in sentences(text):
                self.uni.update(toks)
                self.bi.update(zip(toks, toks[1:]))
                self.tri.update(zip(toks, toks[1:], toks[2:]))
        self.n = sum(self.uni.values())

    def pmi(self, gram: tuple) -> float:
        table = {1: self.uni, 2: self.bi, 3: self.tri}[len(gram)]
        c = table.get(gram if len(gram) > 1 else gram[0], 0)
        if c == 0:
            return float("-inf")
        expected = math.prod(self.uni[t] / self.n for t in gram)
        return math.log2((c / self.n) / expected)

    def npmi(self, gram: tuple) -> float:
        """PMI normalizado em [-1, 1] (Bouma, 2009): menos viesado para n-gramas raros."""
        table = {2: self.bi, 3: self.tri}[len(gram)]
        c = table.get(gram, 0)
        if c == 0:
            return -1.0
        return self.pmi(gram) / -math.log2(c / self.n)

    def collocations(self, order: int = 2, min_count: int = 3) -> list:
        """n-gramas de conteúdo (pontas não são stopwords/pontuação) com contagem mínima."""
        table = {2: self.bi, 3: self.tri}[order]
        out = []
        for gram, c in table.items():
            if c < min_count or not (is_content(gram[0]) and is_content(gram[-1])):
                continue
            if any(not t[0].isalnum() for t in gram):
                continue
            out.append(dict(ngram=" ".join(gram), n=c, pmi=self.pmi(gram), npmi=self.npmi(gram)))
        return out
