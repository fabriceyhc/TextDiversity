# TextDiversity pkgs
import spacy
import numpy as np
from sklearn.decomposition import PCA

from Bio import Align

# locals
from ..metric import TextDiversity
from ..utils import *

# ==================================================================================

class PartOfSpeechSequence(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'verbose': False,
        # PartOfSpeechSequence configs
        'pad_to_max_len': False, 
        'use_gpu': False,
        'n_components': None,
        'split_sentences': False,
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.model = spacy.load("en_core_web_sm")
        self.aligner = Align.PairwiseAligner()

    def align_and_score(self, seq1, seq2):
        return self.aligner.align(seq1, seq2).score

    def extract_features(self, corpus, return_ids=False):

        # clean corpus
        corpus = clean_text(corpus)

        # split sentences
        if self.config['split_sentences']:
            corpus, text_ids, sentence_ids = split_sentences(corpus, return_ids=True)
        else:
            ids = list(range(len(corpus)))
            text_ids, sentence_ids = ids, ids

        # # remove any blanks...
        # corpus = [d for d in corpus if len(d.strip()) > 0]

        # extracts parts-of-speech (poses)
        poses = []
        for s in corpus:
            pos = [token.pos_ for token in self.model(s)] #if token.text not in stopwords]
            poses.append(pos)

        # compute max seq length for padding / normalization later
        self.max_len = find_max_list(poses)

        # pad to max sentence doc length
        if self.config['pad_to_max_len']:
            poses = np.array([s + ['NULL'] * (self.max_len - len(s)) for s in poses])

        if return_ids:
            return poses, corpus, text_ids, sentence_ids 
        return poses, corpus

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
        Z = compute_Z_pairwise(
            features, 
            self.align_and_score, 
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
            score = self.align_and_score(q_feat, f)
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
    print_div_metric(PartOfSpeechSequence, lo_div, hi_div)

    # similarities
    print("similarities")
    print_sim_metric(PartOfSpeechSequence, lo_div, hi_div)

    # rank similarities
    print("rankings")
    print_ranking(PartOfSpeechSequence, ["I was wrong"], lo_div + hi_div)

    # (textdiv) ~\GitHub\TextDiversity\src>python -m textdiversity.text_diversities.morphological