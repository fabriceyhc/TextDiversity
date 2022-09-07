import numpy as np
from ..text_diversities import (
    DocumentSemanticDiversity,
    POSSequenceDiversity,
    RhythmicDiversity,
    DependencyDiversity,
)

class TextFetch:
    def __init__(self, texts: list) -> None:
        self.texts = texts

        # featurizers
        self.semantic_featurizer = DocumentSemanticDiversity()
        self.syntactic_featurizer = DependencyDiversity()
        self.morphological_featurizer = POSSequenceDiversity()
        self.phonological_featurizer = RhythmicDiversity()

        # pre-computed features
        self.semantic_features = None
        self.syntactic_features = None
        self.morphological_features = None
        self.phonological_features = None

        # pre-computed text parses
        self.semantic_text = None
        self.syntactic_text = None
        self.morphological_text = None
        self.phonological_text = None

    def compute_semantic_features(self) -> None:
        features, texts = self.semantic_featurizer.extract_features(self.texts)
        self.semantic_features = features
        self.semantic_text = texts

    def compute_syntactic_features(self) -> None:
        features, texts = self.syntactic_featurizer.extract_features(self.texts)
        self.syntactic_features = features
        self.syntactic_text = texts

    def compute_morphological_features(self) -> None:
        features, texts = self.morphological_featurizer.extract_features(self.texts)
        self.morphological_features = features
        self.morphological_text = texts

    def compute_phonological_features(self) -> None:
        features, texts = self.phonological_featurizer.extract_features(self.texts)
        self.phonological_features = features
        self.phonological_features = texts

    def compute_features(self) -> None:
        self.compute_semantic_features()
        self.compute_syntactic_features()
        self.compute_morphological_features()
        # self.compute_phonological_features()


    def search(self, query: str, linguistic_type: str = "semantic", top_n: int = 1):
        if "semantic" in linguistic_type:
            ranker = self.semantic_featurizer
            t_feats = self.semantic_features
            t_texts = self.semantic_text
        elif "syntactic" in linguistic_type:
            ranker = self.syntactic_featurizer
            t_feats = self.syntactic_features
            t_texts = self.syntactic_text
        elif "morphological" in linguistic_type:
            ranker = self.morphological_featurizer
            t_feats = self.morphological_features
            t_texts = self.morphological_text
        elif "phonological" in linguistic_type:
            ranker = self.phonological_featurizer
            t_feats = self.phonological_features
            t_texts = self.phonological_text
        else:
            raise ValueError(f"Invalid linguistic_type: {linguistic_type}")

        if top_n == -1:
            top_n = len(t_texts)

        q_feats, q_texts = ranker.extract_features([query])

        z = ranker.calculate_similarity_vector(q_feats[0], t_feats)

        # rank based on similarity
        rank_idx = np.argsort(z)[::-1]

        ranking = np.array(t_texts)[rank_idx].tolist()
        scores = z[rank_idx]

        return ranking[:top_n], scores[:top_n]

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

    linguistic_features = ["semantic", "syntactic", "morphological"] #, "phonological"]

    for lf in linguistic_features:
        print()
        start_time = perf_counter()
        ranking, scores = text_fetcher.search(query, lf, top_n=3)
        print(f"{lf} search ({round(perf_counter() - start_time, 2)}s)")
        for text, score in zip(ranking, scores):
            print(f"score: {round(score, 2)} | text: {text}")

    # (dpml) ~\dpml\lineage\search>python textfetch.py
    # precomputation took 16.57 seconds
    # query: long streaks of hilarious gags in this movie

    # semantic search (0.1s)
    # score: 0.87 | text: rich veins of funny stuff in this movie
    # score: 0.79 | text: more than another `` best man '' clone by weaving a theme throughout this funny film
    # score: 0.77 | text: , this gender-bending comedy is generally quite funny .

    # syntactic search (1.18s)
    # score: 1.0 | text: rich veins of funny stuff in this movie
    # score: 0.4099999964237213 | text: a sour taste in one 's mouth
    # score: 0.4099999964237213 | text: we never feel anything for these characters

    # morphological search (0.19s)
    # score: 1.0 | text: rich veins of funny stuff in this movie
    # score: 0.75 | text: by far the worst movie of the year
    # score: 0.75 | text: sharp edges and a deep vein of sadness
        