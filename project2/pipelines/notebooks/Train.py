import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import re

'''e1_text,e1_type,e2_text,e2_type,intermediate_text,relation_label'''

class CSVReader():
    all_anotations: pd.DataFrame
    entities_relations: pd.DataFrame
    entities: pd.DataFrame

    def __init__(self, csvs_list: list[pd.DataFrame]):
        all_annotatins = pd.concat(csvs_list, ignore_index= True)
        first_entities = all_annotatins[['e1_text', 'e1_type']].rename(columns= {'e1_text': 'data', 'e1_type': 'label'})
        second_entitites = all_annotatins[['e2_text', 'e2_type']].rename(columns= {'e2_text': 'data', 'e2_type': 'label'})

        self.entities = pd.concat([first_entities, second_entitites]).drop_duplicates()
        self.entities_relations = all_annotatins[['e1_type', 'e2_type', 'intermediate_text', 'relation_label']]

class Lemmatizer:
    def __init__(self, tool: str = 'en_core_sci_sm'):
        self.nlp = spacy.load(tool, disable=["parser", "ner"])

    def lemmatize_series(self, series: pd.Series) -> pd.Series:
        texts = series.fillna("").astype(str).tolist()
        
        results = []
        for doc in self.nlp.pipe(texts, batch_size=256):
            lemmas = [token.lemma_.lower() for token in doc if not token.is_punct]
            results.append(" ".join(lemmas))
            
        # Retorna uma pd.Series preservando o indice da Series recebida
        return pd.Series(results, index=series.index, name=f"{series.name}")

class EntityClassifier():
    def __init__(self, ngram_range =(1,5), max_features= 2000, maxiter=1000, seed= 42, C= 0.5):
        self.lemmatizer = Lemmatizer()
        self.idf_tf = TfidfVectorizer(ngram_range= ngram_range, max_features= max_features, sublinear_tf= True, min_df= 2)
        self.model = LogisticRegression(max_iter= maxiter, random_state= seed, class_weight= 'balanced', C= C)

    def fit(self, data: pd.Series, labels: pd.Series):
        data_lem = self.lemmatizer.lemmatize_series(data)
        train_data = self.idf_tf.fit_transform(data_lem)
        self.model.fit(train_data, labels)

        print('EntityClassifier trained successfully.')

    def predict(self, data: pd.Series) -> pd.Series:
        data_lem = self.lemmatizer.lemmatize_series(data)
        processed = self.idf_tf.transform(data_lem)
        prediction = self.model.predict(processed)

        return pd.Series(prediction, index= data.index)

class RelationClassifier():
    def __init__(self, ngram_range = (1,5), max_features= 2000, maxiter= 1000, seed= 42, C=0.5):
        self.lemmatizer = Lemmatizer()
        self.idf_tf = TfidfVectorizer(ngram_range= ngram_range, max_features= max_features, sublinear_tf= True, min_df= 2)
        self.model = LogisticRegression(max_iter= maxiter, random_state= seed, class_weight= 'balanced', C= C)
        self.ohe = None

    def fit(self, e1_type : pd.Series, e2_type: pd.Series, intermediate_text: pd.Series, intermediate_label: pd.Series):
        all_types = pd.concat([e1_type, e2_type]).dropna().unique()
        self.ohe = OneHotEncoder(categories=[all_types, all_types], handle_unknown= 'ignore')

        types_df = pd.DataFrame({'e1_type': e1_type, 'e2_type': e2_type})
        X_types = self.ohe.fit_transform(types_df)
        intermediate_text_lem = self.lemmatizer.lemmatize_series(intermediate_text.fillna(''))
        X_intermediate_text = self.idf_tf.fit_transform(intermediate_text_lem)

        X = hstack([X_types, X_intermediate_text])
        self.model.fit(X, intermediate_label)

        print('RelationClassifier trained successfully.')

    def predict(self, e1_type: pd.Series, e2_type: pd.Series, intermediate_text: pd.Series) -> pd.Series:
        types_df = pd.DataFrame({'e1_type': e1_type, 'e2_type': e2_type})
        X_types = self.ohe.transform(types_df)

        intermediate_text_lem = self.lemmatizer.lemmatize_series(intermediate_text.fillna(''))
        X_intermediate_text = self.idf_tf.transform(intermediate_text_lem)

        X = hstack([X_types, X_intermediate_text])
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=e1_type.index)

    
        