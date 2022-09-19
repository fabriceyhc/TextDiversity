# TextDiversity pkgs
import spacy
import torch
import numpy as np
from sklearn.decomposition import PCA

from Bio import Align

# locals
from ..metric import TextDiversity
from ..utils import *

# ==================================================================================
# helper functions 
# ==================================================================================

def align_and_score(seq1, seq2):
    return Align.PairwiseAligner().align(seq1, seq2).score

# ==================================================================================

class POSSequenceDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'verbose': False,
        # POSSequenceDiversity configs
        'pad_to_max_len': False, 
        'use_gpu': False,
        'n_components': None 
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.model = spacy.load("en_core_web_sm")

    def extract_features(self, corpus, return_ids=False):

        # clean corpus
        corpus = clean_text(corpus)

        # split sentences
        sentences, corpus_ids = split_sentences(corpus, return_ids=True)

        # extracts parts-of-speech (poses)
        poses = []
        for s in sentences:
            pos = [token.pos_ for token in self.model(s)] #if token.text not in stopwords]
            poses.append(pos)

        # compute max seq length for padding / normalization later
        self.max_len = find_max_list(poses)

        # pad to max sentence doc length
        if self.config['pad_to_max_len']:
            poses = np.array([s + ['NULL'] * (self.max_len - len(s)) for s in poses])

        if return_ids:
            return poses, sentences, corpus_ids
        return poses, sentences

    def calculate_similarities(self, features):
        """
        Uses biopython.Bio.Align.PairwiseAligner() to align and score stands of
        morphological DNA (i.e. sequences of pos tags)
        """

        if is_list_of_lists(features):
            # convert pos tags to alphas
            features = tag2alpha(features)
            features = ["".join(pos) for pos in features]

        # compute pairwise alignment + scoring
        Z = compute_pairwise(
            features, 
            align_and_score, 
            self.max_len, 
            verbose=self.config['verbose'])

        # normalize the similarities by max_len
        Z = Z / self.max_len

        return Z

    def calculate_similarity_vector(self, q_feat, c_feat):

        features = [q_feat] + c_feat

        if is_list_of_lists(features):
            # convert pos tags to alphas
            features = tag2alpha(features)
            features = ["".join(pos) for pos in features]

        q_feat = features[0]
        c_feat = features[1:]

        q_len = len(q_feat)
        scores = []
        for f in c_feat:
            score = align_and_score(q_feat, f)
            score /= max(len(f), q_len)
            scores.append(score)

        z = np.array(scores)

        return z

    def calculate_abundance(self, species):
        num_species = len(species)
        p = np.full(num_species, 1 / num_species)
        return p

    def __call__(self, corpus): 
        return super().__call__(corpus)


if __name__ == '__main__':

    # TEST
    lo_div = ['one massive earth', 'an enormous globe', 'the colossal world']
    hi_div = ['basic human right', 'you were right', 'make a right']

    # diversities
    print("diversities")
    print_div_metric(POSSequenceDiversity, lo_div, hi_div)

    # similarities
    print("similarities")
    print_sim_metric(POSSequenceDiversity, lo_div, hi_div)

    # rank similarities
    print("rankings")
    print_ranking(POSSequenceDiversity, ["I was wrong"], lo_div + hi_div)

    # (textdiv) ~\GitHub\TextDiversity\src>python -m textdiversity.text_diversities.morphological