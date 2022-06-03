from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm import tqdm
import numpy as np
import tempfile
import os
import csv
from lexicalrichness import LexicalRichness

# locals
import utils

global_score_cache = {}
similarity2diversity_function = lambda sim_score_list: - np.mean(sim_score_list)


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


class Similarity2DiversityMetric(DiversityMetric):
    """
    Implements the diversity to similarity reduction specified on section 5 in the paper
    (https://arxiv.org/pdf/2004.02990.pdf)
    for any similarity metric.

    config:
        shared with the original similarity metric.

    usage:
        metric = Similarity2DiversityMetric(config, SimilarityMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see CosineSimilarity2Diversity
    """

    def __init__(self, config, similarity_metric_class):
        super().__init__(config)
        assert issubclass(similarity_metric_class, SimilarityMetric)
        self.similarity_metric = similarity_metric_class(config)

    def __call__(self, response_set):
        super().__call__(response_set)

        similarity_list = []
        for i in range(len(response_set)):
            for j in range(i):
                similarity_list.append(self.similarity_metric(response_set[i], response_set[j]))
        diversity_score = similarity2diversity_function(similarity_list)
        return diversity_score


class Similarity2DiversityFromFileMetric(DiversityMetric):
    required_input = 'set_index'  # when reading results from a file, the input is the set index

    default_config = {'input_path': None,
                      'num_sets': -1,
                      'samples_per_set': -1}  # required fields - filled by run files

    def __init__(self, config):
        super().__init__(config)

        # validate input
        self.uint_assert('num_sets')
        self.uint_assert('samples_per_set')
        self.input_path_assert('input_path')

        # define cache
        metric_name = utils.CamleCase2snake_case(type(self).__name__)
        self.config['cache_file'] = os.path.join(tempfile.gettempdir(),
                                                 os.path.basename(self.config['input_path']
                                                                  .replace('.csv',
                                                                           '_{}_scores.tsv'.format(metric_name))))
        self.config['input_tsv'] = os.path.join(tempfile.gettempdir(),
                                                os.path.basename(self.config['input_path']
                                                                 .replace('.csv', '_{}_input.tsv'.format(metric_name))))

    @abstractmethod
    def calc_scores(self):
        # input: input_csv
        # output: save score file (as temp_file)
        pass

    def create_input_tsv(self):
        # reformat input_csv for to a tsv file, as an input for sentence similarity neural models

        out_fields = ['index', 'sentence1_id', 'sentence2_id', 'sentence1', 'sentence2']
        f_in = open(self.config['input_path'], 'r', encoding='utf-8')
        f_out = open(self.config['input_tsv'], 'w', encoding='utf-8')
        reader = csv.DictReader(f_in, dialect='excel')
        writer = csv.DictWriter(f_out, fieldnames=out_fields, dialect='excel-tab')
        writer.writeheader()

        for idx, in_row in enumerate(reader):
            for i in range(self.config['samples_per_set']):
                for j in range(i):
                    writer.writerow({
                        'index': idx,
                        'sentence1_id': i,
                        'sentence2_id': j,
                        'sentence1': in_row['resp_{}'.format(i)],
                        'sentence2': in_row['resp_{}'.format(j)],
                    })

        f_in.close()
        f_out.close()

    def get_similarity_scores(self):

        global global_score_cache  # Here we save the scores in memory for cheaper access

        # fetch or calc scores
        if self.config['cache_file'] in global_score_cache.keys():
            scores = global_score_cache[self.config['cache_file']]
        else:
            if not os.path.isfile(self.config['cache_file']) or self.config.get('ignore_cache', False):
                self.calc_scores()
            with open(self.config['cache_file'], 'r') as cache_f:
                scores = cache_f.read().split('\n')[:-1]
                assert len(scores) == self.config['num_sets'] * \
                       sum([i for i in range(self.config['samples_per_set'])])  # choose(samples_per_set, 2)
                scores = [float(e) for e in scores]
                scores = np.reshape(scores, [self.config['num_sets'], -1])
                global_score_cache[self.config['cache_file']] = scores  # cache
        return scores

    def __call__(self, response_set_idx):

        # validate input
        assert type(response_set_idx) == int

        similarity_list = self.get_similarity_scores()[response_set_idx, :]
        diversity_score = similarity2diversity_function(similarity_list)
        return diversity_score


class AveragedNgramDiversityMetric(DiversityMetric):
    """
    Calculates the mean values of an n-gram based diversity metric in range n \in [n_min, n_max].

    config:
        shared with the original n-gram metric.
        n_min(int) > 0 - Specify the lowest n-gram value to be averaged
        n_max(int) > 0 - Specify the highest n-gram value to be averaged

    usage:
        metric = AveragedNgramDiversityMetric(config, NgramMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see AveragedDistinctNgrams
    """

    def __init__(self, config, ngram_metric_class):
        super().__init__(config)

        # validate config
        self.uint_assert('n_min')
        self.uint_assert('n_max')
        err_msg = 'AveragedNgramMetric config must include n_max > n_min > 0 (int) representing n-gram size.'
        assert self.config['n_max'] > self.config['n_min'] > 0, err_msg

        # add n field
        self.config['n'] = self.config['n_min']

        # instance ngram metric
        assert issubclass(ngram_metric_class, DiversityMetric)
        self.ngram_metric = ngram_metric_class(self.config)

    def __call__(self, response_set):
        super().__call__(response_set)

        ngrams_results = []
        for n in range(self.config['n_min'], self.config['n_max'] + 1):
            self.config['n'] = n
            result = self.ngram_metric(response_set)
            # print('{}, {}'.format(self.ngram_metric.config['n'], result))
            ngrams_results.append(result)
        return np.mean(ngrams_results)

class LexicalScorer(DiversityMetric):
    def __init__(self, config):
        super().__init__(config)
        self.scorer = LexicalRichness
        self.metric = config['metric']
        self.window_size = config['window_size']
        self.threshold = config['threshold']
        self.num_draws = config['num_draws']

    def __call__(self, response_set):
        if type(response_set) == list:
            response_set = ''.join(response_set)

        scores = self.scorer(response_set, use_TextBlob=True)

        if self.metric in ['words', 'terms', 'ttr', 'rttr', 'cttr']:
            return getattr(scores, self.metric)

        elif self.metric in ['msttr', 'mattr']:
            assert type(self.window_size) == int 
            if self.window_size > getattr(scores, 'words'):
                self.window_size = getattr(scores, 'words')
            return getattr(scores, self.metric)(self.window_size)
        elif self.metric in ['mtld']:
            assert type(self.threshold) == float 
            return getattr(scores, self.metric)(self.threshold) 
        elif self.metric in ['hdd']:
            assert type(self.num_draws) == int 
            return getattr(scores, self.metric)(self.num_draws) 
        else:
            raise ValueError("Invalid metric selected. Pick from the following:\n"\
                "['words', 'terms', 'ttr', 'rttr', 'cttr', 'msttr', 'mattr', 'mtld', 'hdd']"
            )
            

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
    def calculate_abundance(self, features):

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