# TextDiversity pkgs
import spacy
import torch
import numpy as np
import pandas as pd
import itertools
from functools import partial
import logging
logging.basicConfig(level=logging.CRITICAL)

from Bio import Align

# rhythmic diversity
import cadence as cd
import string

# phonemic diversity
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

import os
import sys
from collections import Counter

# locals
from ..metric import TextDiversity
from ..utils import *

# ==================================================================================
# helper functions 
# ==================================================================================

def align_and_score(seq1, seq2):
    return Align.PairwiseAligner().align(seq1, seq2).score

# ==================================================================================

class RhythmicDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'remove_stopwords': False, 
        'verbose': False,
        # RhythmicDiversity configs
        'pad_to_max_len': False
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

        # strip punctuation
        sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]

        # remove any blank sentences...
        sentences = [s for s in sentences if len(s.strip()) > 0]

        # extracts rhythms (sequences of [un]weighted [un]stressed syllables)
        rhythms = []
        for s in sentences:
            prose = cd.Prose(s)
            df = prose.sylls().reset_index()
            if all([c in df.columns for c in ['syll_stress', 'syll_weight']]):
                rhythm = df[df['word_ipa_i'] == 1][['syll_stress', 'syll_weight']].values
                rhythm = ["".join(row) for row in rhythm]
            else:
                rhythm = [""]
            rhythms.append(rhythm)

        # compute max seq length for padding / normalization later
        self.max_len = find_max_list(rhythms)

        # pad to max sentence doc length
        if self.config['pad_to_max_len']:
            rhythms = np.array([r + ['N'] * (self.max_len - len(r)) for r in rhythms])

        if return_ids:
            return rhythms, sentences, corpus_ids
        return rhythms, sentences

    def calculate_similarities(self, features):
        """
        Uses biopython.Bio.Align.PairwiseAligner() to align and score stands of
        phonological DNA (i.e. sequences of [un]weighted [un]stressed syllables)
        """

        if is_list_of_lists(features):
            # convert rhythm tags to alphas
            features = tag2alpha(features)
            features = ["".join(rhythm) for rhythm in features]

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
            # convert rhythm tags to alphas
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


class PhonemicDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'remove_stopwords': False, 
        'verbose': False,
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)

        # load pre-computed phonemic similarity matrix Z
        file_dir = os.path.dirname(__file__)
        csv_path = os.path.join(file_dir, '../similarities/phoneme_similarities.csv')
        self.Z = pd.read_csv(csv_path, index_col=0)

        # phonemizer
        logger = logging.getLogger()
        logger.disabled = True
        self.backend = EspeakBackend('en-us', words_mismatch='ignore', logger=logger)
        self.separator = Separator(phone=' ', word=None)

    def get_phonemes(self, corpus, return_counts=True):
        # remove all the punctuation from the text, considering only the specified
        # punctuation marks
        text = Punctuation(';:,.!"?()-/><').remove(corpus)

        # build the set of all the words in the text
        words = {w.lower() for line in text for w in line.strip().split(' ') if w}

        # build the lexicon by phonemizing each word one by one. The backend.phonemize
        # function expect a list as input and outputs a list.
        lexicon = {
            word : self.backend.phonemize([word], separator=self.separator, strip=True)[0]
            for word in words} 

        if return_counts:
            phoneme_counts = Counter(" ".join(lexicon.values()).split(" "))
            del phoneme_counts[" "]
            return lexicon, phoneme_counts
        
        return lexicon 

    def extract_features(self, corpus):

        # get phonemes
        lexicon, phoneme_counts = self.get_phonemes(corpus)

        # TODO: Think of a more elegant way of doing this...
        #       The phonemic diversity calculation doesn't follow
        #       the same flow as other diversities because we
        #       precomputed the Z matrix and the main thing 
        #       needed now is the phoneme_counts.
        return phoneme_counts, phoneme_counts

    def calculate_similarities(self, features):

        unwanted = set(features) - set(self.Z)
        for unwanted_key in unwanted: del features[unwanted_key]
        phoneme_list = list(features.keys())
        Z = self.Z.loc[phoneme_list][phoneme_list].to_numpy()

        return Z

    def calculate_similarity_vector(self, q_feat, c_feat):
        raise Exception("Ranking requires metrics that operate on the document level. Try RhythmicDiversity instead.")

    def calculate_abundance(self, species):
        p = np.array(list(species.values()), dtype=np.float16)
        p /= p.sum()
        return p

    def __call__(self, corpus): 
        return super().__call__(corpus)


if __name__ == '__main__':

    # TEST
    lo_div = ['one massive earth', 'an enormous globe', 'the colossal world']
    hi_div = ['basic human right', 'you were right', 'make a right']

    # diversities
    print("diversities")
    print_div_metric(RhythmicDiversity, lo_div, hi_div)
    print_div_metric(PhonemicDiversity, lo_div, hi_div)

    # similarities
    print("similarities")
    print_sim_metric(RhythmicDiversity, lo_div, hi_div)
    print_sim_metric(PhonemicDiversity, lo_div, hi_div)

    # rank similarities
    print("rankings")
    print_ranking(RhythmicDiversity, ["I was wrong"], lo_div + hi_div)
    # print_ranking(PhonemicDiversity, "burn big planets", lo_div + hi_div)

    # (textdiv) ~\GitHub\TextDiversity\src>python -m textdiversity.text_diversities.phonological