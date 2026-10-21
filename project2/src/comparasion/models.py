"""Modelos de Regressão Logística (n-gramas) e integrações do pipeline de entidades/relações."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from . import config, data

class ManualLRRelationModel:
    """Modelo de Regressão Logística com TF-IDF (word + char n-grams) treinado nas anotações manuais.
    
    Suporta dois modos:
    1. Apenas texto intermediário (para comparação direta com MLP e P1)
    2. Texto intermediário + Tipos das Entidades (modos com tipos Gold e Tipos Preditos)
    """
    
    def __init__(self, use_entity_types: bool = False):
        self.use_entity_types = use_entity_types
        self.vectorizer = FeatureUnion([
            ('word_ngram', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
            ('char_ngram', TfidfVectorizer(ngram_range=(2, 5), analyzer='char', max_features=5000))
        ])
        self.clf = LogisticRegression(max_iter=1000, random_state=config.SEED, class_weight='balanced')
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.Series:
        inter = df["intermediate_text"].fillna("").astype(str)
        if self.use_entity_types:
            e1_t = df["e1_type"].fillna("").astype(str)
            e2_t = df["e2_type"].fillna("").astype(str)
            return e1_t + " " + inter + " " + e2_t
        return inter

    def fit(self, train_dfs: list[pd.DataFrame]):
        train_df = pd.concat(train_dfs, ignore_index=True)
        # Filtrar apenas dados com relações válidas para o treino
        train_df = train_df[train_df["relation_label"].isin(config.REAL_RELATIONS)]
        
        X_text = self._prepare_features(train_df)
        y = train_df["relation_label"]
        
        X_vec = self.vectorizer.fit_transform(X_text)
        self.clf.fit(X_vec, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X_text = self._prepare_features(df)
        X_vec = self.vectorizer.transform(X_text)
        preds = self.clf.predict(X_vec)
        return pd.Series(preds, index=df.index)

def train_and_eval_lr(gold_df: pd.DataFrame, use_types: bool = False) -> pd.Series:
    """Carrega dados manuais, treina Regressão Logística e realiza predição sobre o Gold."""
    train_dfs = data.load_manual_annotations()
    lr = ManualLRRelationModel(use_entity_types=use_types)
    lr.fit(train_dfs)
    return lr.predict(gold_df)