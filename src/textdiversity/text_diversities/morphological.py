# TextDiversity pkgs
import spacy
import torch
import numpy as np
import itertools
from functools import partial
from sklearn.decomposition import PCA

from Bio import Align

# locals
import metric
from utils import *

# ==================================================================================
# helper functions 
# ==================================================================================

def align_and_score(seq1, seq2):
    return Align.PairwiseAligner().align(seq1, seq2).score

# ==================================================================================

class POSSequenceDiversity(metric.TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'verbose': False,
        # POSSequenceDiversity configs
        'pad_to_max_len': False, 
        'pos_to_alpha' : True,
        'use_gpu': False,
        'n_components': None 
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.model = spacy.load("en_core_web_sm")

    def extract_features(self, corpus):

        # clean corpus
        corpus = clean_text(corpus)

        # split sentences
        sentences = split_sentences(corpus)

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

        # convert poses to alpha
        if self.config['pos_to_alpha']:
            # build dict of unique poses
            pos_map = set(itertools.chain(*poses))
            pos_map = {tag: chr(i+65) for i, tag in enumerate(pos_map)}
            # convert to int for distance comparison
            if isinstance(poses, np.ndarray):
                pos_to_alpha_fn = np.vectorize(pos_map.get)
                poses = pos_to_alpha_fn(poses)
            else:
                poses = [list(map(pos_map.get, pos)) for pos in poses]

        # create strands of morphological dna
        pos_strands = ["".join(pos) for pos in poses]

        return pos_strands, sentences

    def calculate_similarities(self, features):
        """
        Uses biopython.Bio.Align.PairwiseAligner() to align and score stands of
        morphological DNA (i.e. sequences of pos tags)
        """

        # compute pairwise alignment + scoring
        Z = compute_pairwise(
            features, 
            align_and_score, 
            self.max_len, 
            verbose=self.config['verbose'])

        # normalize the similarities by max_len
        Z = Z / self.max_len

        return Z

    def calculate_abundance(self, species):
        num_species = len(species)
        p = np.full(num_species, 1 / num_species)
        return p

    def __call__(self, corpus): 
        return super().__call__(corpus)


if __name__ == '__main__':

    def print_metric(metric_class, lo_div, hi_div):

        div_metric = metric_class({'normalize': False})
        div_lo_unnorm = div_metric(lo_div)
        div_hi_unnorm = div_metric(hi_div)

        div_metric.config['normalize'] = True
        div_lo_norm = div_metric(lo_div)
        div_hi_norm = div_metric(hi_div)
        
        print('metric:{0} | lo_div: {1:0.3f} ({2:0.3f}) | hi_div: {3:0.3f} ({4:0.3f}) | passed: {5}'
            .format(
                metric_class.__name__,
                div_lo_unnorm, 
                div_lo_norm, 
                div_hi_unnorm,
                div_hi_norm,
                div_lo_unnorm < div_hi_unnorm 
            and div_lo_norm < div_hi_norm))

    # TEST
    lo_div = ['one massive earth', 'an enormous globe', 'the colossal world']
    hi_div = ['basic human right', 'you were right', 'make a right']

    print_metric(POSSequenceDiversity, lo_div, hi_div)