import numpy as np
import scipy.linalg as la
import pdb

from transformations.diverse_paraphrase.submod.submodular_funcs import (
    distinct_ngrams,
    ngram_overlap,
    similarity_func,
    seq_func,
    ngram_overlap_unit,
    similarity_gain,
    seq_gain,
    # textdiversity
    div_helper,
    sim_helper
)

from textdiversity import (
    TokenSemanticDiversity
    DocumentSemanticDiversity
    POSSequenceDiversity
    RhythmicDiversity
    PhonemicDiversity
    DependencyDiversity
)


class SubmodularOpt:
    """
    A class used to select final candidates for diverse paraphrasing
    using submodular optimization

    """
    def __init__(self, V=None, v=None, **kwargs):
        """
        Parameters
        ---

        V : list of str
            Ground Set Generations from which candidates are selected
        v : str
            Sentence which is used to select semantically equivalent
            outputs
        """
        self.v = v
        self.V = V

        # similarity functions
        self.toksim_fn = TokenSemanticDiversity()
        self.docsim_fn = DocumentSemanticDiversity()

        # diversity functions
        self.posdiv_fn = POSSequenceDiversity()
        self.rhydiv_fn = RhythmicDiversity()
        self.phodiv_fn = PhonemicDiversity()
        self.depdiv_fn = DependencyDiversity()

    def initialize_function(self, 
                            lam, 
                            w_toksim = 1.0,
                            w_docsim = 1.0,
                            w_posdiv = 1.0, 
                            w_rhydiv = 1.0, 
                            w_phodiv = 1.0, 
                            w_depdiv = 1.0):
        """
        Parameters
        ---

        lam: float (0 <= lam <= 1.)
            Determines fraction of weight assigned to the diversity and similarity components
        w_toksim : float
            Weight assigned to token semantic similarity between V and v.
        w_docsim : float
            Weight assigned to document semantic similarity between V and v.
        w_posdiv : float
            Weight assigned to part-of-speech (pos) sequence diversity.
        w_rhydiv : float
            Weight assigned to rhythmic diversity.
        w_phodiv : float
            Weight assigned to phonemic diversity.
        w_posdiv : float
            Weight assigned to syntactical diversity via dependency parses.
        """
        
        self.lam = lam

        # similarity weights
        self.w_toksim = w_toksim
        self.w_docsim = w_docsim

        # diversity weights
        self.w_posdiv = w_posdiv
        self.w_rhydiv = w_rhydiv
        self.w_phodiv = w_phodiv
        self.w_posdiv = w_posdiv

    def final_func(self, pos_sets, rem_list, selec_set, normalize=False):
        
        toksim_scores, docsim_scores = [], []
        posdiv_scores, rhydiv_scores, phodiv_scores, depdiv_scores = [], [], [], []
        
        for doc in rem_list:

            # similarities
            toksim_scores.append(sim_helper(doc, self.v, self.toksim_fn, normalize))
            docsim_scores.append(sim_helper(doc, self.v, self.docsim_fn, normalize))
            
            # diversities
            posdiv_scores.append(div_helper(doc, self.v, self.posdiv_fn, normalize))
            rhydiv_scores.append(div_helper(doc, self.v, self.rhydiv_fn, normalize))
            phodiv_scores.append(div_helper(doc, self.v, self.phodiv_fn, normalize))
            depdiv_scores.append(div_helper(doc, self.v, self.depdiv_fn, normalize))

        sim_score = self.w_toksim * np.array(toksim_scores) 
                  + self.w_docsim * np.array(docsim_scores)
        div_score = self.w_posdiv * np.array(posdiv_scores) 
                  + self.w_rhydiv * np.array(rhydiv_scores) 
                  + self.w_phodiv * np.array(phodiv_scores) 
                  + self.w_posdiv * np.array(depdiv_scores) 

        final_score = self.lam * sim_score + (1 - self.lam) * div_score

        return final_score

    def maximize_func(self, k=5):
        selec_sents = set()
        ground_set = set(self.V)
        selec_set = set(selec_sents)
        rem_set = ground_set.difference(selec_set)
        while len(selec_sents) < k:

            rem_list = list(rem_set)
            pos_sets = [list(selec_set.union({x})) for x in rem_list]

            score_map = self.final_func(pos_sets, rem_list, selec_set)
            max_idx = np.argmax(score_map)

            selec_sents = pos_sets[max_idx]
            selec_set = set(selec_sents)
            rem_set = ground_set.difference(selec_set)

        return selec_sents
