import re
import spacy


class MedicalTextProcessor:
    def __init__(self, spacy_model: str = "en_core_sci_sm"):
        # Mantém parser e tagger ativos (úteis para sents e noun_chunks), desativa apenas o NER
        self.nlp = spacy.load(spacy_model, disable=["ner"])

    def clean_text(self, text: str) -> str:
        """Limpa ruídos típicos de textos médicos (citações, figuras, colchetes e espaços múltiplos)."""
        # 1. Remove referências a figuras/tabelas ex: (Fig 1), (Table 2)
        text = re.sub(r'\s*\((?:Fig|Figure|Table|Figs)\.?\s*\d+[-–\d]*\)', '', text, flags=re.IGNORECASE)
        
        # 2. Remove referências numéricas em colchetes ex: [1], [12-15]
        text = re.sub(r'\[\d+[-–\d,]*\]', '', text)
        
        # 3. Normaliza hífens e travessões
        text = text.replace('–', '-').replace('—', '-')
        
        # 4. Normaliza espaços múltiplos e quebras de linha
        return re.sub(r'\s+', ' ', text).strip()

    def split_sentences(self, raw_text: str) -> list[str]:
        """Limpa o texto e o divide em períodos/frases usando o parser sintático do spaCy."""
        cleaned_text = self.clean_text(raw_text)
        doc = self.nlp(cleaned_text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

    def extract_entity_candidates(self, doc) -> list[dict]:
        """Extrai candidatos a entidades usando Sintagmas Nominais (Noun Chunks)."""
        candidates = []
        for chunk in doc.noun_chunks:
            clean_chunk = chunk.text.strip()
            if len(clean_chunk) > 2:
                candidates.append({
                    'text': clean_chunk,
                    'start_token': chunk.start,
                    'end_token': chunk.end,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                })
        return candidates

    def extract_intermediate_text(self, doc, e1: dict, e2: dict, max_distance: int = 10, context_k: int = 3) -> str | None:
        """
        Gera o texto intermediário resumido entre duas entidades dentro do mesmo período.
        Retorna None se as entidades estiverem fora de ordem ou muito distantes.
        """
        token_distance = e2['start_token'] - e1['end_token']

        # Descarta se e2 vem antes de e1 ou se há sobreposição
        if token_distance < 0:
            return None

        # Descarta se a distância em tokens ultrapassar o limite máximo
        if token_distance > max_distance:
            return None

        tokens_between = doc[e1['end_token']:e2['start_token']]

        # Se o trecho intermediário for curto, utiliza integralmente
        if len(tokens_between) <= (context_k * 2):
            return tokens_between.text.strip()

        # Se for longo, extrai apenas os k primeiros e k últimos tokens do trecho
        head_context = tokens_between[:context_k].text
        tail_context = tokens_between[-context_k:].text

        return f"{head_context} {tail_context}".strip()

    def process_sentence(self, sentence_text: str, max_distance: int = 10, context_k: int = 3):
        """
        Processa um único período e retorna:
        - doc: objeto spacy.tokens.Doc
        - candidates: lista de candidatos a entidades (sintagmas nominais)
        - pairs: lista de pares candidatos com seus respectivos textos intermediários válidos
        """
        doc = self.nlp(sentence_text)
        candidates = self.extract_entity_candidates(doc)
        
        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                e1 = candidates[i]
                e2 = candidates[j]
                
                intermed_text = self.extract_intermediate_text(
                    doc, e1, e2, max_distance=max_distance, context_k=context_k
                )
                
                if intermed_text is not None:
                    pairs.append({
                        'e1': e1,
                        'e2': e2,
                        'intermediate_text': intermed_text
                    })
                    
        return doc, candidates, pairs