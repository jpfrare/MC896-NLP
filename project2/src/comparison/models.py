"""Treina e aplica todos os modelos de ENTIDADE e RELAÇÃO sobre o gold, de forma uniforme."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from . import config, data, p1_baseline

NGRAM_ABLATION = [(1, 1), (1, 2), (1, 3), (1, 5)]

# nome -> (família, fonte de rótulos, representação)
RELATION_INFO = {
    "Projeto 1 (regras)": ("Regras (verbos/distância)", "Regras escritas à mão", "Padrões lexicais"),
    "Baseline (classe majoritária)": ("Dummy", "Anotação manual", "-"),
    "LR n-gramas [manual]": ("Regressão Logística", "Anotação manual (12 docs)", "TF-IDF 1-5gramas + tipos (gold)"),
    "LR n-gramas [manual, tipos previstos]": ("Regressão Logística", "Anotação manual (12 docs)", "TF-IDF 1-5gramas + tipos previstos pelo EntityClassifier"),
    "LR n-gramas [weak sup.]": ("Regressão Logística", "Regras do P1 (weak supervision)", "TF-IDF 1-5gramas + tipos"),
    "MLP TF-IDF [weak sup.]": ("MLP (PyTorch)", "Regras do P1 (weak supervision)", "TF-IDF 1-2gramas (só texto)"),
    "Naive Bayes [manual]": ("Naive Bayes", "Anotação manual (12 docs)", "TF-IDF 1-5gramas + tipos"),
    "SVM linear [manual]": ("SVM", "Anotação manual (12 docs)", "TF-IDF 1-5gramas + tipos"),
    "Random Forest [manual]": ("Random Forest", "Anotação manual (12 docs)", "TF-IDF 1-5gramas + tipos"),
}


class _SparseFeatures:
    """Mesmas features do RelationClassifier dos colegas (TF-IDF do contexto + tipos), reutilizável."""

    def __init__(self, Train, ngram_range=(1, 5), use_types=True):
        self.use_types = use_types
        self.lem = Train.Lemmatizer()
        self.tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=2000, sublinear_tf=True, min_df=2)
        self.ohe = None

    def fit_transform(self, df):
        if not self.use_types:
            return self.tfidf.fit_transform(self.lem.lemmatize_series(df["intermediate_text"])).tocsr()
        types = pd.concat([df["e1_type"], df["e2_type"]]).dropna().unique()
        self.ohe = OneHotEncoder(categories=[types, types], handle_unknown="ignore")
        xt = self.ohe.fit_transform(df[["e1_type", "e2_type"]])
        return hstack([xt, self.tfidf.fit_transform(self.lem.lemmatize_series(df["intermediate_text"]))]).tocsr()

    def transform(self, df):
        if not self.use_types:
            return self.tfidf.transform(self.lem.lemmatize_series(df["intermediate_text"])).tocsr()
        xt = self.ohe.transform(df[["e1_type", "e2_type"]])
        return hstack([xt, self.tfidf.transform(self.lem.lemmatize_series(df["intermediate_text"]))]).tocsr()


def _fit_lr_relation(Train, df, seed, ngram_range=(1, 5)):
    rc = Train.RelationClassifier(ngram_range=ngram_range, C=0.1, seed=seed)
    rc.fit(df["e1_type"], df["e2_type"], df["intermediate_text"], df["relation_label"])
    return rc


def relation_predictions(gold, include_mlp=True, include_extras=True, seed=config.SEED, log=print):
    """Retorna (DataFrame de predições por modelo, dict de metadados, dict de avisos)."""
    Train = data.import_colleague_code()
    manual = pd.concat(data.load_manual_annotations(), ignore_index=True)
    weak = data.load_weak_training(exclude_docs=gold["doc_id"].unique())
    log(f"  treino manual: {len(manual)} pares | weak supervision: {len(weak)} pares | gold: {len(gold)} pares")

    preds, notes = {}, {}
    preds["Projeto 1 (regras)"] = p1_baseline.p1_relation_predictions(gold, data.load_p1_edges())
    preds["Baseline (classe majoritária)"] = pd.Series(manual["relation_label"].mode()[0], index=gold.index)

    log("  treinando LR n-gramas [manual] ...")
    rc = _fit_lr_relation(Train, manual, seed)
    notes["lemmatizer"] = rc.lemmatizer.backend
    preds["LR n-gramas [manual]"] = rc.predict(gold["e1_type"], gold["e2_type"], gold["intermediate_text"])

    # Variante SEM oráculo: os tipos das entidades vêm do EntityClassifier (pipeline mais realista)
    reader = Train.CSVReader(data.load_manual_annotations())
    ec = Train.EntityClassifier(C=0.1, seed=seed)
    ec.fit(reader.entities["data"], reader.entities["label"])
    t1, t2 = ec.predict(gold["e1_text"]), ec.predict(gold["e2_text"])
    preds["LR n-gramas [manual, tipos previstos]"] = rc.predict(t1, t2, gold["intermediate_text"])

    log("  treinando LR n-gramas [weak sup.] ...")
    rcw = _fit_lr_relation(Train, weak, seed)
    preds["LR n-gramas [weak sup.]"] = rcw.predict(gold["e1_type"], gold["e2_type"], gold["intermediate_text"])

    if include_mlp:
        try:
            from .mlp_torch import WeakMLPRelationModel
            log("  treinando MLP [weak sup.] (PyTorch) ...")
            mlp = WeakMLPRelationModel(seed=seed).fit(weak["intermediate_text"], weak["relation_label"])
            preds["MLP TF-IDF [weak sup.]"] = mlp.predict(gold["intermediate_text"])
            notes["mlp"] = f"{mlp.epochs_run} épocas, melhor loss de treino {mlp.best_loss:.4f}, classes {mlp.classes_}"
        except ImportError:
            notes["mlp"] = "PULADO: PyTorch não instalado (pip install torch)"
            log("  [aviso] torch não encontrado: MLP pulado.")

    if include_extras:
        log("  treinando Naive Bayes / SVM / Random Forest [manual] ...")
        feats = _SparseFeatures(Train)
        Xtr, ytr = feats.fit_transform(manual), manual["relation_label"].to_numpy()
        Xte = feats.transform(gold)
        for name, clf in [
            ("Naive Bayes [manual]", MultinomialNB(alpha=0.1)),
            ("SVM linear [manual]", LinearSVC(C=0.5, class_weight="balanced", random_state=seed)),
            ("Random Forest [manual]", RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]:
            preds[name] = pd.Series(clf.fit(Xtr, ytr).predict(Xte), index=gold.index)

    df = pd.DataFrame(preds)
    info = {k: v for k, v in RELATION_INFO.items() if k in df.columns}
    return df, info, notes


def ngram_ablation(gold, seed=config.SEED, log=print):
    """Possibilidade 5 do guia: granularidade dos n-gramas (LR, anotação manual).

    Duas configurações: 'com tipos' (texto + tipos do gold) e 'só texto' (contribuição pura dos n-gramas).
    Retorna {(granularidade, configuração): predições}.
    """
    from sklearn.linear_model import LogisticRegression
    Train = data.import_colleague_code()
    manual = pd.concat(data.load_manual_annotations(), ignore_index=True)
    names = {(1, 1): "unigramas", (1, 2): "uni+bigramas", (1, 3): "uni+bi+trigramas", (1, 5): "1 a 5-gramas"}
    out = {}
    for ng in NGRAM_ABLATION:
        for use_types, setting in ((True, "com tipos"), (False, "só texto")):
            log(f"  ablação n-gramas {ng} | {setting} ...")
            feats = _SparseFeatures(Train, ng, use_types)
            clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced", C=0.1)
            clf.fit(feats.fit_transform(manual), manual["relation_label"])
            out[(names[ng], setting)] = pd.Series(clf.predict(feats.transform(gold)), index=gold.index)
    return out


def entity_predictions(entities, seed=config.SEED, log=print):
    """Projeto 1 (nós oficiais) x EntityClassifier (LR + n-gramas) nas menções do gold."""
    Train = data.import_colleague_code()
    reader = Train.CSVReader(data.load_manual_annotations())
    log(f"  treinando EntityClassifier em {len(reader.entities)} menções únicas ...")
    ec = Train.EntityClassifier(C=0.1, seed=seed)
    ec.fit(reader.entities["data"], reader.entities["label"])
    p1_pred, coverage = p1_baseline.p1_entity_predictions(entities, data.load_p1_nodes())
    lr_pred = ec.predict(entities["text"])
    maj = pd.Series(reader.entities["label"].mode()[0], index=entities.index)
    df = pd.DataFrame({"Projeto 1 (regras)": p1_pred.values,
                       "LR n-gramas [manual]": lr_pred.values,
                       "Baseline (classe majoritária)": maj.values}, index=entities.index)
    return df, coverage
