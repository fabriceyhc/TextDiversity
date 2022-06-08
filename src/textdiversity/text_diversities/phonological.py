# TextDiversity pkgs
import spacy
import torch
import numpy as np
import pandas as pd
import itertools
from functools import partial

from Bio import Align

# rhythmic diversity
import cadence as cd

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

def get_phonemes(corpus, return_counts=True):
    # remove all the punctuation from the text, considering only the specified
    # punctuation marks
    text = Punctuation(';:,.!"?()-/><').remove(corpus)

    # build the set of all the words in the text
    words = {w.lower() for line in text for w in line.strip().split(' ') if w}

    # initialize the espeak backend for English
    backend = EspeakBackend('en-us', words_mismatch='ignore')

    # separate phones by a space and ignoring words boundaries
    separator = Separator(phone=' ', word=None)

    # build the lexicon by phonemizing each word one by one. The backend.phonemize
    # function expect a list as input and outputs a list.
    lexicon = {
        word : backend.phonemize([word], separator=separator, strip=True)[0]
        for word in words} 

    if return_counts:
        phoneme_counts = Counter(" ".join(lexicon.values()).split(" "))
        del phoneme_counts[" "]
        return lexicon, phoneme_counts
    
    return lexicon 

# ==================================================================================

class RhythmicDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'remove_stopwords': False, 
        'verbose': False,
        # RhythmicDiversity configs
        'pad_to_max_len': False, 
        'rhythm_to_alpha' : True
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

        # extracts rhythms (sequences of [un]weighted [un]stressed syllables)
        rhythms = []
        for s in sentences:
            prose = cd.Prose(s)
            df = prose.sylls().reset_index()
            rhythm = df[df['word_ipa_i'] == 1][['syll_stress', 'syll_weight']].values
            rhythm = ["".join(row) for row in rhythm]
            rhythms.append(rhythm)

        # compute max seq length for padding / normalization later
        self.max_len = find_max_list(rhythms)

        # pad to max sentence doc length
        if self.config['pad_to_max_len']:
            rhythms = np.array([r + ['N'] * (self.max_len - len(r)) for r in rhythms])

        # convert rhythms to alpha
        if self.config['rhythm_to_alpha']:
            # build dict of unique rhythms
            rhythm_map = set(itertools.chain(*rhythms))
            rhythm_map = {tag: chr(i+65) for i, tag in enumerate(rhythm_map)}
            # convert to int for distance comparison
            if isinstance(rhythm, np.ndarray):
                rhythm_to_alpha_fn = np.vectorize(rhythm_map.get)
                rhythm = rhythm_to_alpha_fn(rhythm)
            else:
                rhythms = [list(map(rhythm_map.get, r)) for r in rhythms]

        # create strands of rhythmic dna
        rhythm_strands = ["".join(r) for r in rhythms]

        return rhythm_strands, sentences

    def calculate_similarities(self, features):
        """
        Uses biopython.Bio.Align.PairwiseAligner() to align and score stands of
        morphological DNA (i.e. sequences of [un]weighted [un]stressed syllables)
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

    def extract_features(self, corpus):

        # get phonemes
        lexicon, phoneme_counts = get_phonemes(corpus)

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