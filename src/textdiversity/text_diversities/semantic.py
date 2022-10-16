# TextDiversity pkgs
import itertools
from itertools import chain

import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

import transformers
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from gensim.utils import tokenize
import gensim.downloader

from nltk.corpus import stopwords

import amrlib

# locals
from ..similarities.amrsim import WLKScorer
from ..metric import TextDiversity
from ..utils import *

# config
transformers.logging.set_verbosity_error()

# ==================================================================================
# helper functions 
# ==================================================================================

# NA

# ==================================================================================


class TokenSemanticDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'distance_fn': distance.chebyshev, 
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'remove_punct': False,
        'scale_dist': "exp", 
        'power_reg': False, 
        'mean_adj': True,
        'verbose': False,
        # TokenSemanticDiversity configs
        'MODEL_NAME': "bert-base-uncased", # "roberta-base", "microsoft/deberta-large", # "bert-base-uncased", # "facebook/bart-large-cnn",
        'batch_size': 16,
        'use_gpu': True,
        'n_components': None 
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config['MODEL_NAME'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
        self.undesirable_tokens = [
            self.tokenizer.pad_token_id, 
            self.tokenizer.cls_token_id, 
            self.tokenizer.sep_token_id
        ]
        self.batch_size = config['batch_size']
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')

        # move model to device
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)

        # print('{0} ({1})'.format(self.__class__.__name__, config['MODEL_NAME']))

    def encode(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # emb = out.last_hidden_state
        emb = out.hidden_states[-2]
        # emb = torch.stack(out.hidden_states[-4:]).mean(axis=0)
        # emb = torch.stack(out.hidden_states[-4:]).sum(axis=0)
        # emb = torch.cat(out.hidden_states[-4:], dim=-1)
        return emb

    def extract_features(self, corpus):
        inputs = self.tokenizer(corpus, return_tensors='pt', padding=True, truncation=True)
        batches = zip(chunker(inputs.input_ids, self.batch_size), 
                      chunker(inputs.attention_mask, self.batch_size))
        if self.config['verbose']:
            print('getting token embeddings...')
            batches = tqdm(batches, total=int(len(inputs.input_ids)/self.batch_size))

        outputs = []
        for input_ids, attention_mask in batches:
            emb = self.encode(input_ids.to(self.device), 
                       attention_mask.to(self.device))
            outputs.append(emb)
        embeddings = torch.cat(outputs)

        # remove undesirable tokens
        idx = np.isin(inputs['input_ids'],  self.undesirable_tokens, assume_unique=True, invert=True).reshape(-1)
        # idx = np.isin(inputs['input_ids'],  [self.tokenizer.cls_token_id], assume_unique=True).reshape(-1)
        tok = np.array(self.tokenizer.convert_ids_to_tokens(inputs.input_ids.view(-1)))[idx]
        boe = embeddings.view(-1, embeddings.shape[-1])[idx].detach().cpu().numpy()

        # remove stopwords
        if self.config['remove_stopwords']:
            idx = np.isin(tok, stopwords.words('english'), invert=True)
            tok = tok[idx]
            boe = boe[idx]

        # remove punctuation
        if self.config['remove_punct']:
            punct = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
            punct = [c for c in punct]
            idx = np.isin(tok, punct, invert=True)
            tok = tok[idx]
            boe = boe[idx]

        # compress embedding to speed up similarity matrix computation
        if self.config['n_components'] == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.config['verbose']:
                print('Using n_components={}'.format(str(n_components)))
        else:
            n_components = -1

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = self.config['dim_reducer'](n_components=n_components).fit_transform(boe)

        if len(np.flatnonzero(np.core.defchararray.find(tok,'##')!=-1)) > 0:
            tok, boe = merge_bpe(tok, boe)

        return boe, tok

    # def calculate_similarities(self, features):
    #     Z = compute_pairwise(features, self.config['distance_fn'])
    #     Z = Z - Z.mean()
    #     Z = 1 / (1+np.e**Z)
    #     np.fill_diagonal(Z, 1)
    #     return Z

    def calculate_similarities(self, features):
        num_embeddings = len(features)

        Z = np.eye(num_embeddings)
        iu = np.triu_indices(num_embeddings, k=1)
        il = (iu[1], iu[0])

        iterable = range(num_embeddings)
        if self.config['verbose']:
            print('calculating similarity matrix...')
            iterable = tqdm(iterable)

        for e1 in iterable:
            for e2 in range(1, num_embeddings - e1):
                d = self.config['distance_fn'](features[e1], features[e1 + e2])
                if self.config['scale_dist'] == "exp":
                    d = np.exp(-d)
                elif self.config['scale_dist'] == "invert":
                    d = 1 - d
                Z[e1][e1 + e2] = d     
        Z[il] = Z[iu]

        # remove some noise from the Z similarities
        if self.config['power_reg']:
            Z **= 2

        # remove some noise from the Z similarities
        if self.config['mean_adj']:
            off_diag = np.where(~np.eye(Z.shape[0],dtype=bool))
            Z[off_diag] -= Z[off_diag].mean()
            Z = np.where(Z < 0, 0 , Z)

        return Z

    def calculate_similarity_vector(self, q_feat, c_feat):
        raise Exception("Ranking requires metrics that operate on the document level. Try DocumentSemanticDiversity instead.")


    def calculate_abundance(self, species):
        num_species = len(species)
        p = np.full(num_species, 1 / num_species)
        return p

    def __call__(self, response_set): 
        return super().__call__(response_set)


class TokenEmbeddingDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'distance_fn': distance.chebyshev, 
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'remove_punct': False,
        'scale_dist': "exp", 
        'power_reg': False, 
        'mean_adj': True,
        'verbose': False,
        # TokenEmbeddingDiversity configs
        'EMBEDDING': 'word2vec-google-news-300', # glove-wiki-gigaword-300, fasttext-wiki-news-subwords-300, 'word2vec-google-news-300'
        'batch_size': 16,
        'n_components': None  
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)

        self.model = gensim.downloader.load(config['EMBEDDING'])
        self.batch_size = config['batch_size']

        # print('{0} ({1})'.format(self.__class__.__name__, config['EMBEDDING']))


    def extract_features(self, corpus):

        # tokenize
        tok = []
        for s in corpus:
            tok.append(list(tokenize(s, lowercase=True)))
        tok = list(itertools.chain(*tok))
        tok = [t for t in tok if t in self.model.index_to_key]

        # embed
        embeddings = [self.model[t] for t in tok]

        tok = np.array(tok)
        boe = np.stack(embeddings)

        # remove stopwords
        if self.config['remove_stopwords']:
            idx = np.isin(tok, stopwords.words('english'), invert=True)
            tok = tok[idx]
            boe = boe[idx]

        # remove punctuation
        if self.config['remove_punct']:
            punct = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
            punct = [c for c in punct]
            idx = np.isin(tok, punct, invert=True)
            tok = tok[idx]
            boe = boe[idx]

        # compress embedding to speed up similarity matrix computation
        if self.config['n_components'] == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.config['verbose']:
                print('Using n_components={}'.format(str(n_components)))
        else:
            n_components = -1

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = self.config['dim_reducer'](n_components=n_components).fit_transform(boe)

        if len(np.flatnonzero(np.core.defchararray.find(tok,'##')!=-1)) > 0:
            tok, boe = merge_bpe(tok, boe)

        return boe, tok

    def calculate_similarities(self, features):
        num_embeddings = len(features)

        Z = np.eye(num_embeddings)
        iu = np.triu_indices(num_embeddings, k=1)
        il = (iu[1], iu[0])

        iterable = range(num_embeddings)
        if self.config['verbose']:
            print('calculating similarity matrix...')
            iterable = tqdm(iterable)

        for e1 in iterable:
            for e2 in range(1, num_embeddings - e1):
                d = self.config['distance_fn'](features[e1], features[e1 + e2])
                if self.config['scale_dist'] == "exp":
                    d = np.exp(-d)
                elif self.config['scale_dist'] == "invert":
                    d = 1 - d
                Z[e1][e1 + e2] = d     
        Z[il] = Z[iu]

        # remove some noise from the Z similarities
        if self.config['power_reg']:
            Z **= 2

        # remove some noise from the Z similarities
        if self.config['mean_adj']:
            off_diag = np.where(~np.eye(Z.shape[0],dtype=bool))
            Z[off_diag] -= Z[off_diag].mean()
            Z = np.where(Z < 0, 0 , Z)

        return Z


    def calculate_similarity_vector(self, q_feat, c_feat):
        raise Exception("Ranking requires metrics that operate on the document level. Try DocumentSemanticDiversity instead.")


    def calculate_abundance(self, species):
        num_species = len(species)
        p = np.full(num_species, 1 / num_species)
        return p

    def __call__(self, response_set): 
        return super().__call__(response_set)


class STokenSemanticDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'distance_fn': distance.cosine, 
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'remove_punct': True,
        'scale_dist': "invert", 
        'power_reg': False, 
        'mean_adj': False,
        'verbose': False,
        # SentenceSemanticDiversity configs
        'MODEL_NAME':"bert-large-nli-stsb-mean-tokens",
        'use_gpu': True,
        'n_components': None 
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(config['MODEL_NAME'], device=self.device)
        self.config['verbose'] = config['verbose']
        self.undesirable_tokens = [
            self.model.tokenizer.pad_token_id, 
            self.model.tokenizer.cls_token_id, 
            self.model.tokenizer.sep_token_id
        ]

        # print('{0} ({1})'.format(self.__class__.__name__, config['MODEL_NAME']))

    def extract_features(self, corpus):

        inputs = self.model.tokenizer(corpus)
        tok = list(chain.from_iterable(inputs.input_ids))

        embeddings = self.model.encode(corpus, output_value='token_embeddings')

        boe = torch.cat(embeddings).numpy()   

        # remove undesirable tokens
        idx = np.isin(tok, self.undesirable_tokens, assume_unique=True, invert=True).reshape(-1)
        # idx = np.isin(inputs['input_ids'],  [self.tokenizer.cls_token_id], assume_unique=True).reshape(-1)
        tok = np.array(self.model.tokenizer.convert_ids_to_tokens(torch.tensor(tok).view(-1)))

        assert len(boe) == len(tok), "boe.shape: {}\n tok.shape: {}".format(boe.shape, tok.shape)

        tok = tok[idx]
        boe = boe[idx]

        # remove stopwords
        if self.config['remove_stopwords']:
            idx = np.isin(tok, stopwords.words('english'), invert=True)
            tok = tok[idx]
            boe = boe[idx]

        # remove punctuation
        if self.config['remove_punct']:
            punct = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
            punct = [c for c in punct]
            idx = np.isin(tok, punct, invert=True)
            tok = tok[idx]
            boe = boe[idx]
        
        # compress embedding to speed up similarity matrix computation
        if self.config['n_components'] == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.config['verbose']:
                print('Using n_components={}'.format(str(n_components)))
        else:
            n_components = -1

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = self.config['dim_reducer'](n_components=n_components).fit_transform(boe)

        if len(np.flatnonzero(np.core.defchararray.find(tok,'##')!=-1)) > 0:
            tok, boe = merge_bpe(tok, boe)

        return boe, tok

    def calculate_similarities(self, features):
        num_embeddings = len(features)

        Z = np.eye(num_embeddings)
        iu = np.triu_indices(num_embeddings, k=1)
        il = (iu[1], iu[0])

        iterable = range(num_embeddings)
        if self.config['verbose']:
            print('calculating similarity matrix...')
            iterable = tqdm(iterable)

        for e1 in iterable:
            for e2 in range(1, num_embeddings - e1):
                d = self.config['distance_fn'](features[e1], features[e1 + e2])
                if self.config['scale_dist'] == "exp":
                    d = np.exp(-d)
                elif self.config['scale_dist'] == "invert":
                    d = 1 - d
                Z[e1][e1 + e2] = d     
        Z[il] = Z[iu]

        # remove some noise from the Z similarities
        if self.config['power_reg']:
            Z **= 2

        # remove some noise from the Z similarities
        if self.config['mean_adj']:
            off_diag = np.where(~np.eye(Z.shape[0],dtype=bool))
            Z[off_diag] -= Z[off_diag].mean()
            Z = np.where(Z < 0, 0 , Z)

        return Z

    def calculate_similarity_vector(self, q_feat, c_feat):
        raise Exception("Ranking requires metrics that operate on the document level. Try DocumentSemanticDiversity instead.")

    def calculate_abundance(self, species):
        num_species = len(species)
        p = np.full(num_species, 1 / num_species)
        return p

    def __call__(self, response_set): 
        return super().__call__(response_set)


class DocumentSemanticDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'distance_fn': distance.cosine, 
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'scale_dist': "invert", 
        'power_reg': False, 
        'mean_adj': False,
        'verbose': False,
        # DocumentSemanticDiversity configs
        'MODEL_NAME': "princeton-nlp/sup-simcse-roberta-large", # "bert-large-nli-stsb-mean-tokens",
        'use_gpu': True,
        'n_components': None
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(config['MODEL_NAME'], device=self.device)
        self.config['verbose'] = config['verbose']

        # print('{0} ({1})'.format(self.__class__.__name__, config['MODEL_NAME']))

    def extract_features(self, corpus, return_ids=False):

        boe = np.stack(self.model.encode(corpus))
        
        # compress embedding to speed up similarity matrix computation
        if self.config['n_components'] == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.config['verbose']:
                print('Using n_components={}'.format(str(n_components)))
        else:
            n_components = -1

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = self.config['dim_reducer'](n_components=n_components).fit_transform(boe)

        if return_ids:
            return boe, corpus, list(range(len(boe)))
        return boe, corpus

    def calculate_similarities(self, features):
        num_embeddings = len(features)

        Z = np.eye(num_embeddings)
        iu = np.triu_indices(num_embeddings, k=1)
        il = (iu[1], iu[0])

        iterable = range(num_embeddings)
        if self.config['verbose']:
            print('calculating similarity matrix...')
            iterable = tqdm(iterable)

        for e1 in iterable:
            for e2 in range(1, num_embeddings - e1):
                d = self.config['distance_fn'](features[e1], features[e1 + e2])
                if self.config['scale_dist'] == "exp":
                    d = np.exp(-d)
                elif self.config['scale_dist'] == "invert":
                    d = 1 - d
                Z[e1][e1 + e2] = d     
        Z[il] = Z[iu]

        # remove some noise from the Z similarities
        if self.config['power_reg']:
            Z **= 2

        # remove some noise from the Z similarities
        if self.config['mean_adj']:
            off_diag = np.where(~np.eye(Z.shape[0],dtype=bool))
            Z[off_diag] -= Z[off_diag].mean()
            Z = np.where(Z < 0, 0 , Z)

        return Z


    def calculate_similarity_vector(self, q_feat, c_feat):

        z = np.array([self.config['distance_fn'](q_feat, f) for f in c_feat])
        
        if self.config['scale_dist'] == "exp":
            z = np.exp(-z) 
        elif self.config['scale_dist'] == "invert":
            z = 1 - z

        # remove some noise from the z similarities
        if self.config['power_reg']:
            z **= 2

        if self.config['mean_adj']:
            z -= z.mean()
            z = np.where(z < 0, 0 , z)

        return z

    def calculate_abundance(self, species):
        num_species = len(species)
        p = np.full(num_species, 1 / num_species)
        return p

    def __call__(self, response_set): 
        return super().__call__(response_set)


class AMRDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'verbose': False,
        # AMRDiversity configs
        'use_gpu': True,
        'batch_size': 4
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.config['verbose'] = config['verbose']
        
        # make sure stog model is downloaded 
        # to amrlib's package repo
        download_stog_model()
        self.model = amrlib.load_stog_model(device=self.device, batch_size=self.config['batch_size'])
        self.scorer = WLKScorer().compute_score

        # print('{0} ({1})'.format(self.__class__.__name__, config['MODEL_NAME']))

    def extract_features(self, corpus, return_ids=False):
        graphs = self.model.parse_sents(corpus, add_metadata=False)
        if return_ids:
            return graphs, corpus, list(range(len(corpus)))
        return graphs, corpus

    def calculate_similarities(self, features):
        num_embeddings = len(features)

        Z = np.eye(num_embeddings)
        iu = np.triu_indices(num_embeddings, k=1)
        il = (iu[1], iu[0])

        iterable = range(num_embeddings - 1)
        if self.config['verbose']:
            print('calculating similarity matrix...')
            iterable = tqdm(iterable)

        # for e1 in iterable:
        #     for e2 in range(1, num_embeddings - e1):
        #         s = self.scorer([features[e1]], [features[e1 + e2]])[0]
        #         Z[e1][e1 + e2] = s

        for e1 in iterable:
            other_idx = e1 + 1
            c_feat = features[other_idx:]
            q_feat = [features[e1]] * len(c_feat)
            s = self.scorer(q_feat, c_feat)
            Z[e1][other_idx:] = s
        Z[il] = Z[iu]
        return Z

    def calculate_similarity_vector(self, q_feat, c_feat):
        # z = np.array([self.scorer([q_feat], [f])[0] for f in c_feat])
        z = self.scorer([q_feat] * len(c_feat), c_feat)
        return z

    def calculate_abundance(self, species):
        num_species = len(species)
        p = np.full(num_species, 1 / num_species)
        return p

    def __call__(self, response_set): 
        return super().__call__(response_set)



if __name__ == '__main__':

    # TEST
    lo_div = ['one massive earth', 'an enormous globe', 'the colossal world']
    hi_div = ['basic human right', 'you were right', 'make a right']

    # diversities
    print("diversities")
    print_div_metric(TokenSemanticDiversity, lo_div, hi_div)
    print_div_metric(TokenEmbeddingDiversity, lo_div, hi_div)
    print_div_metric(STokenSemanticDiversity, lo_div, hi_div)
    print_div_metric(DocumentSemanticDiversity, lo_div, hi_div)
    print_div_metric(AMRDiversity, lo_div, hi_div)

    # similarities
    print("similarities")
    print_sim_metric(TokenSemanticDiversity, lo_div, hi_div)
    print_sim_metric(TokenEmbeddingDiversity, lo_div, hi_div)
    print_sim_metric(STokenSemanticDiversity, lo_div, hi_div)
    print_sim_metric(DocumentSemanticDiversity, lo_div, hi_div)
    print_sim_metric(AMRDiversity, lo_div, hi_div)

    # rank similarities
    print("rankings")
    print_ranking(DocumentSemanticDiversity, ["a big planet"], lo_div + hi_div)
    print_ranking(AMRDiversity, ["a big planet"], lo_div + hi_div)

    # (textdiv) ~\GitHub\TextDiversity\src>python -m textdiversity.text_diversities.semantic