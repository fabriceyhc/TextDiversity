from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm import tqdm
import numpy as np
import tempfile
import os
import csv

class Metric(ABC):
    use_me = False  # static var indicates to run files whether or not to use this metric
    default_config = {}  # static var, specifies the default config for run files

    def __init__(self, config):
        self.config = config

        # validate config
        assert type(self.config) == dict, 'Metric config must be dict type.'

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def uint_assert(self, field_name):
        err_msg = 'Required: {}(int) > 0'.format(field_name)
        assert type(self.config.get(field_name, None)) == int, err_msg
        assert self.config[field_name] > 0, err_msg

    def input_path_assert(self, field_name):
        err_msg = '[{}] not exists.'.format(field_name)
        assert os.path.exists(self.config.get(field_name, None)), err_msg


class DiversityMetric(Metric):
    required_input = 'response_set'  # in most cases, the diversity metric input is the response set S_c

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def __call__(self, response_set):
        # validate input
        assert type(response_set) == list
        assert all([type(e) == str for e in response_set])

        # place holder
        diversity_score = None
        return diversity_score


class SimilarityMetric(Metric):

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def __call__(self, resp_a, resp_b):
        # validate input
        assert type(resp_a) == type(resp_b) == str

        # place holder
        similarity_score = None
        return similarity_score
        

class TextDiversity(DiversityMetric):

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, corpus):

        if not all([type(d) == str and d.strip() != "" for d in corpus]):
            print('corpus contains invalid inputs: \n {} \n returning 0'.format(corpus))
            return 0

        # extract features + species
        features, species = self.extract_features(corpus)

        # if there are no features, diversity is 0 by default
        if len(features) == 0:
            return 0

        # get similarity matrix Z
        Z = self.calculate_similarities(features)
        
        # get abundance vector p
        p = self.calculate_abundance(species)

        # calculate diversity
        D = self.calc_div(p, Z, self.config['q'])

        # optionally normalize diversity by number of species
        # which is the length of the p vector
        if self.config['normalize']:
            D /= len(p)

        return D

    def similarity(self, corpus):

        if not all([type(d) == str and d.strip() != "" for d in corpus]):
            print('corpus contains invalid inputs: \n {} \n returning 0'.format(corpus))
            return 0

        # extract features + species
        features, species = self.extract_features(corpus)

        # if there are no features, similarity is 0 by default
        if len(features) == 0:
            return 0

        # get similarity matrix Z
        Z = self.calculate_similarities(features)

        # calculate similarity score
        similarity = Z.sum() / (len(Z) ** 2)
        
        return similarity

    def rank_similarity(self, query, corpus, top_n=1):

        if top_n == -1:
            top_n = len(corpus)

        # extract features + species
        feats, corpus = self.extract_features(query + corpus)
        q_feats, q_corpus = feats[0], corpus[0]
        c_feats, c_corpus = feats[1:], corpus[1:]

        # print(list(zip([q_corpus], [q_feats])))
        # print(list(zip(c_corpus, c_feats)))

        # if there are no features, we cannot rank
        if len(q_feats) == 0 or len(c_feats) == 0:
            return [], []

        # get similarity vector z
        z = self.calculate_similarity_vector(q_feats, c_feats)

        # rank based on similarity
        rank_idx = np.argsort(z)[::-1]

        ranking = np.array(c_corpus)[rank_idx].tolist()
        scores = z[rank_idx]

        return ranking[:top_n], scores[:top_n]


    @abstractmethod
    def extract_features(self, corpus, *args, **kwargs):  
        # validate input
        assert type(corpus) == list
        assert all([type(d) == str for d in corpus])

        # place holders
        features, species = None, None
        return features, species 

    @abstractmethod
    def calculate_similarities(self, features):

        # place holder
        Z = None
        return Z

    @abstractmethod
    def calculate_abundance(self, species):

        # place holder
        p = None
        return p
        
        # num_embeddings = len(embs)

        # Z = np.eye(num_embeddings)
        # iu = np.triu_indices(num_embeddings, k=1)
        # il = (iu[1], iu[0])

        # iterable = range(num_embeddings)
        # if self.verbose:
        #     print('calculating similarity matrix...')
        #     iterable = tqdm(iterable)

        # for e1 in iterable:
        #     for e2 in range(1, num_embeddings - e1):
        #         d = self.config['distance_fn'](embs[e1], embs[e1 + e2])
        #         if self.config['scale_dist'] == "exp":
        #             d = np.exp(-d)
        #         elif self.config['scale_dist'] == "invert":
        #             d = 1 - d
        #         Z[e1][e1 + e2] = d     
        # Z[il] = Z[iu]

        # # remove some noise from the Z similarities
        # if self.config['sq_reg']:
        #     Z **= 2 

        # # remove some noise from the Z similarities
        # if self.config['mean_adj']:
        #     off_diag = np.where(~np.eye(Z.shape[0],dtype=bool))
        #     Z[off_diag] -= Z[off_diag].mean()
        #     Z = np.where(Z < 0, 0 , Z)

        # return Z

    def calc_div(self, p, Z, q=1):
        Zp =  Z @ p
        if q == 1:
            D = 1 / np.prod(Zp ** p)
        elif q == float('inf'):
            D = 1 / Zp.max()
        else:
            D = (p * Zp ** (q-1)).sum() ** (1/(1-q))
        return D  

    def diversity_profile(self, response_set, range=None):

        # embed inputs
        embs, species = self.get_embeddings(response_set)

        if len(embs) == 0:
            return 0

        # get similarity matrix 
        Z = self.calc_Z(embs)

        # get diversities for all q in range
        if range is None:
            range = np.arange(0, 101)

        num_species = len(species)
        p = np.full(num_species, 1/num_species)

        Ds = []
        for q in range:
            D = self.calc_div(p, Z, q)
            Ds.append(D)

        # plot diversity profile
        ax = sns.lineplot(x=range, y=Ds)  
        ax.set(xlabel="Sensitivity Parameter, $q$", 
               ylabel="Diversity $^qD^{\mathbf{Z}}(\mathbf{p})$", 
               title="Corpus Diversity Profile")
        plt.show()

    def species_heatmap(self, response_set, n=10):

        # embed inputs
        embs, species = self.get_embeddings(response_set)

        if len(embs) == 0:
            return 0

        # get similarity matrix 
        Z = self.calc_Z(embs)

        plt.figure(figsize = (10,8))

        g = sns.heatmap(
            data=np.around(Z[:n,:n], 2), 
            xticklabels=species[:n],
            yticklabels=species[:n],
            vmin=0,
            vmax=1,
            annot=True, 
            annot_kws={"fontsize": 10}, 
            fmt='g')
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.set_title('Similarities')
        plt.show()