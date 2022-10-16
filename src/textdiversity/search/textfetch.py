import numpy as np
from collections import defaultdict 
from ..text_diversities import (
    DocumentSemanticDiversity,
    AMRDiversity,
    POSSequenceDiversity,
    RhythmicDiversity,
    DependencyDiversity,
)

class TextFetch:
    def __init__(self, texts: list) -> None:
        self.texts = texts
        self.text_ids = []

        # featurizers
        self.semantic_featurizer = DocumentSemanticDiversity()
        self.amr_featurizer = AMRDiversity()
        self.syntactic_featurizer = DependencyDiversity()
        self.morphological_featurizer = POSSequenceDiversity()
        self.phonological_featurizer = RhythmicDiversity()

        # pre-computed features
        self.semantic_features = None
        self.amr_features = None
        self.syntactic_features = None
        self.morphological_features = None
        self.phonological_features = None

        # pre-computed text parses
        self.semantic_text = None
        self.amr_text = None
        self.syntactic_text = None
        self.morphological_text = None
        self.phonological_text = None

    def compute_semantic_features(self) -> None:
        features, texts, text_ids = self.semantic_featurizer.extract_features(self.texts, return_ids=True)
        self.semantic_features = features
        self.semantic_text = texts
        self.semantic_text_ids = text_ids

    def compute_amr_features(self) -> None:
        features, texts, text_ids = self.amr_featurizer.extract_features(self.texts, return_ids=True)
        self.amr_features = features
        self.amr_text = texts
        self.amr_text_ids = text_ids

    def compute_syntactic_features(self) -> None:
        features, texts, text_ids = self.syntactic_featurizer.extract_features(self.texts, return_ids=True)
        self.syntactic_features = features
        self.syntactic_text = texts
        self.syntactic_text_ids = text_ids

    def compute_morphological_features(self) -> None:
        features, texts, text_ids = self.morphological_featurizer.extract_features(self.texts, return_ids=True)
        self.morphological_features = features
        self.morphological_text = texts
        self.morphological_text_ids = text_ids

    def compute_phonological_features(self) -> None:
        features, texts, text_ids = self.phonological_featurizer.extract_features(self.texts, return_ids=True)
        self.phonological_features = features
        self.phonological_features = texts
        self.phonological_text_ids = text_ids

    def compute_features(self) -> None:
        self.compute_semantic_features()
        self.compute_amr_features()
        self.compute_syntactic_features()
        self.compute_morphological_features()
        # self.compute_phonological_features()

    def get_linguistic_data(self, linguistic_type: str):
        if "semantic" in linguistic_type:
            ranker  = self.semantic_featurizer
            t_feats = self.semantic_features
            t_texts = self.semantic_text
            t_ids   = self.semantic_text_ids
        elif "amr" in linguistic_type:
            ranker  = self.amr_featurizer
            t_feats = self.amr_features
            t_texts = self.amr_text
            t_ids   = self.amr_text_ids
        elif "syntactic" in linguistic_type:
            ranker  = self.syntactic_featurizer
            t_feats = self.syntactic_features
            t_texts = self.syntactic_text
            t_ids   = self.syntactic_text_ids
        elif "morphological" in linguistic_type:
            ranker  = self.morphological_featurizer
            t_feats = self.morphological_features
            t_texts = self.morphological_text
            t_ids   = self.morphological_text_ids
        elif "phonological" in linguistic_type:
            ranker  = self.phonological_featurizer
            t_feats = self.phonological_features
            t_texts = self.phonological_text
            t_ids   = self.phonological_text_ids
        else:
            raise ValueError(f"Invalid linguistic_type: {linguistic_type}")

        return ranker, t_feats, t_texts, np.array(t_ids, dtype=np.int32)

    def search_for_text(self, query: str, linguistic_type: str = "semantic", top_n: int = 1):

        ranker, t_feats, t_texts, t_ids = self.get_linguistic_data(linguistic_type)

        if top_n == -1:
            top_n = len(t_texts)

        q_feats, q_texts = ranker.extract_features([query])

        z = ranker.calculate_similarity_vector(q_feats[0], t_feats)

        # rank based on similarity
        rank_idx = np.argsort(z)[::-1]
        t_ids = t_ids[rank_idx]

        ranking = np.array(t_texts)[rank_idx].tolist()
        scores = z[rank_idx]

        return ranking[:top_n], scores[:top_n]

    def search_for_ids(self, query: str, linguistic_type: str = "semantic", top_n: int = 1):

        ranker, t_feats, t_texts, corpus_ids = self.get_linguistic_data(linguistic_type)
        
        corpus_tuples = [(corpus_id, sentence_id) for sentence_id, corpus_id in enumerate(corpus_ids)]
        corpus_map = defaultdict(list)
        for (corpus_id, sentence_id) in corpus_tuples:
            corpus_map[corpus_id].append(sentence_id)

        if top_n == -1:
            top_n = len(t_texts)

        q_feats, q_texts = ranker.extract_features([query])

        z = ranker.calculate_similarity_vector(q_feats[0], t_feats)

        # rank based on similarity
        rank_idx = np.argsort(z)[::-1]
        t_ids = np.array(corpus_ids)[rank_idx]
        corpus_ids = list(dict.fromkeys(t_ids))
        corpus_scores = [z[corpus_map[id]].mean() for id in corpus_ids]

        return corpus_ids[:top_n], corpus_scores[:top_n]

if __name__ == "__main__":

    import numpy as np
    from time import perf_counter
    from datasets import load_dataset

    dataset = load_dataset("glue", "sst2", split="train[:1000]")
    dataset = dataset.rename_column("sentence", "text")

    text_fetcher = TextFetch(dataset['text'])

    start_time = perf_counter()
    text_fetcher.compute_features()
    print(f"precomputation took {round(perf_counter() - start_time, 2)} seconds")

    query = "long streaks of hilarious gags in this movie"

    print(f"query: {query}")

    linguistic_features = ["semantic", "amr", "syntactic", "morphological"] #, "phonological"]

    print("search_for_text")
    for lf in linguistic_features:
        print()
        start_time = perf_counter()
        ranking, scores = text_fetcher.search_for_text(query, lf, top_n=3)
        print(f"{lf} search ({round(perf_counter() - start_time, 2)}s)")
        for text, score in zip(ranking, scores):
            print(f"score: {round(score, 2)} | text: {text}")

    print("search_for_ids")
    for lf in linguistic_features:
        print()
        start_time = perf_counter()
        corpus_ids, scores = text_fetcher.search_for_ids(query, lf, top_n=3)
        print(f"{lf} search ({round(perf_counter() - start_time, 2)}s)")
        for id, score in zip(corpus_ids, scores):
            print(f"score: {round(score, 2)} | id: {id} | text: {dataset['text'][id]}")  

    # (textdiv) (textdiv) ~\TextDiversity\src>python -m textdiversity.search.textfetch
    # precomputation took 203.26 seconds
    # query: long streaks of hilarious gags in this movie
    # search_for_text

    # semantic search (0.07s)
    # score: 0.87 | text: rich veins of funny stuff in this movie 
    # score: 0.79 | text: more than another `` best man '' clone by weaving a theme throughout this funny film 
    # score: 0.77 | text: , this gender-bending comedy is generally quite funny . 

    # amr search (1.41s)
    # score: 0.3 | text: love this movie 
    # score: 0.29 | text: rich veins of funny stuff in this movie 
    # score: 0.28 | text: in this wildly uneven movie 

    # syntactic search (0.12s)
    # score: 1.0 | text: rich veins of funny stuff in this movie
    # score: 0.41 | text: a sour taste in one 's mouth
    # score: 0.41 | text: we never feel anything for these characters

    # morphological search (0.09s)
    # score: 1.0 | text: rich veins of funny stuff in this movie
    # score: 0.75 | text: sharp edges and a deep vein of sadness
    # score: 0.75 | text: by far the worst movie of the year
    # search_for_ids

    # semantic search (0.06s)
    # score: 0.87 | id: 60 | text: rich veins of funny stuff in this movie 
    # score: 0.79 | id: 35 | text: more than another `` best man '' clone by weaving a theme throughout this funny film 
    # score: 0.77 | id: 105 | text: , this gender-bending comedy is generally quite funny . 

    # amr search (1.38s)
    # score: 0.3 | id: 698 | text: love this movie 
    # score: 0.29 | id: 60 | text: rich veins of funny stuff in this movie 
    # score: 0.28 | id: 181 | text: in this wildly uneven movie 

    # syntactic search (0.13s)
    # score: 1.0 | id: 60 | text: rich veins of funny stuff in this movie 
    # score: 0.41 | id: 554 | text: a sour taste in one 's mouth 
    # score: 0.41 | id: 66 | text: we never feel anything for these characters 
        