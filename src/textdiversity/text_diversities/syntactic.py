# TextDiversity pkgs
import spacy
import torch
import numpy as np
from functools import partial
from sklearn.decomposition import PCA

# tree libs
import networkx as nx
from karateclub import FeatherGraph, GL2Vec, Graph2Vec, LDP
import zss

# # graphviz has compatibility issues on windows
# from networkx.drawing.nx_pydot import graphviz_layout
# from matplotlib import pyplot as plt

# locals
from ..metric import TextDiversity
from ..utils import *

# ==================================================================================
# helper functions 
# ==================================================================================

# def plot_graph(G):
#     pos = nx.nx_agraph.pygraphviz_layout(G, prog="dot")
#     num_nodes = G.number_of_nodes()
#     # fig = plt.figure(1, figsize=(max(12, num_nodes), max(6, num_nodes/2)), dpi=60)
#     nx.draw_networkx(G, pos, with_labels=False, node_color='#00b4d9')
#     # draw node labels
#     node_labels = nx.get_node_attributes(G, 'text') 
#     nx.draw_networkx_labels(G, pos, node_labels)
#     # draw edge labels
#     edge_labels = nx.get_edge_attributes(G,'dep')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels)
#     plt.show()

# used with nx.graph_edit_distance() ===============================================
def node_match_on_pos(G1_node, G2_node):
    return G1_node['pos'] == G2_node['pos']

def edge_match_on_dep(G1_edge, G2_edge):
    return G1_edge['dep'] == G2_edge['dep']

# used with zss ====================================================================
def get_nodes_dict(T):
    nodes_dict = {}
    for edge in T.edges():
        if edge[0] not in nodes_dict:
            nodes_dict[edge[0]] = zss.Node(edge[0])
        if edge[1] not in nodes_dict:
            nodes_dict[edge[1]] = zss.Node(edge[1])
        nodes_dict[edge[0]].addkid(nodes_dict[edge[1]])
    return nodes_dict

def zss_tree_edit_distance(G1, G2):
    source1 = [n for (n, d) in G1.in_degree() if d == 0][0]
    T1 = nx.dfs_tree(G1, source=source1)
    T1_nodes_dict = get_nodes_dict(T1)
    
    source2 = [n for (n, d) in G2.in_degree() if d == 0][0]
    T2 = nx.dfs_tree(G2, source=source2)
    T2_nodes_dict = get_nodes_dict(T2)

    return zss.simple_distance(T1_nodes_dict[source1], T2_nodes_dict[source2])

# ==================================================================================

class DependencyDiversity(TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'verbose': False,
        # DependencyDiversity configs
        'similarity_type':"ldp",
        'use_gpu': False,
        'n_components': None 
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.model = spacy.load("en_core_web_sm")

    def generate_dependency_tree(self, in_string):
        ''' 
        NOTES: 
          - Use the index instead of the token to avoid loops
          - Ensure the node is a string since zss tree edit distance requires it
        '''
        doc = self.model(in_string)

        G = nx.DiGraph()

        nodes = [(str(token.i), {'text': token.text, 'pos' : token.pos_}) 
                 for token in doc 
                 # if not token.is_punct
        ]
        G.add_nodes_from(nodes)

        edges = [(str(token.head.i), str(token.i), {'dep' : token.dep_}) 
                 for token in doc 
                 if token.head.i != token.i 
                 # and not token.is_punct
                 # and not token.head.is_punct
        ]
        G.add_edges_from(edges)

        return G

    def extract_features(self, corpus, return_ids=False):
        """
        self.similarity_type: 
          - "graph_edit_distance" --> nx.graph_edit_distance
          - "tree_edit_distance" --> zss.simple_distance
          - "ldp" --> karateclub.LDP (Local Degree Profile)
          - "feather" --> karateclub.FeatherGraph
        """

        # clean corpus
        corpus = clean_text(corpus)

        # split sentences
        sentences, corpus_ids = split_sentences(corpus, return_ids=True)

        # generate dependency tree graphs
        features = [self.generate_dependency_tree(s) for s in sentences]

        # optionally embed graphs
        if 'distance' not in self.config['similarity_type']:
        
            # the embedding approaches require integer node labels
            features = [nx.convert_node_labels_to_integers(g, first_label=0, ordering='default') for g in features]

            if self.config['similarity_type'] == "ldp":
                model = LDP(bins=64) # more bins, less similarity
                model.fit(features)
                emb = model.get_embedding().astype(np.float32)
            elif self.config['similarity_type'] == "feather":
                model = FeatherGraph(theta_max=100) # higher theta, less similarity
                model.fit(features)
                emb = model.get_embedding().astype(np.float32)

            # compress embedding to speed up similarity matrix computation
            if self.config['n_components'] == "auto":
                n_components = min(max(2, len(emb) // 10), emb.shape[-1])
                if self.config['verbose']:
                    print('Using n_components={}'.format(str(n_components)))
            else:
                n_components = -1

            if type(n_components) == int and n_components > 0 and len(emb) > 1:
                emb = self.config['dim_reducer'](n_components=n_components).fit_transform(emb)

            features = emb

        if return_ids:
            return features, sentences, corpus_ids
        return features, sentences

    def calculate_similarities(self, features):
        """
        self.similarity_type: 
          - "graph_edit_distance" --> nx.graph_edit_distance
          - "tree_edit_distance" --> zss.simple_distance
          - "ldp" --> karateclub.LDP (Local Degree Profile)
          - "feather" --> karateclub.FeatherGraph
        """

        if 'distance' in self.config['similarity_type']:

            if self.config['similarity_type'] == "graph_edit_distance":
                dist_fn = partial(nx.graph_edit_distance, 
                             node_match=node_match_on_pos, 
                             edge_match=edge_match_on_dep)
            elif self.config['similarity_type'] == "tree_edit_distance":
                dist_fn = zss_tree_edit_distance

            Z = compute_pairwise(features, dist_fn)

            # convert distance to similarity

            # # option 1
            # Z = 1 - (Z / (Z.max()))
            # Z **= 4
            # np.fill_diagonal(Z, 1)

            # option 2
            Z = Z - Z.mean()
            Z = 1 / (1+np.e**Z)
            np.fill_diagonal(Z, 1)

        else:

            Z = cos_sim(features, features).numpy()

            # strongly penalize for any differences to make Z more intuitive
            Z **= 200

        return Z

    def calculate_similarity_vector(self, q_feat, c_feat):

        if 'distance' in self.config['similarity_type']:

            if self.config['similarity_type'] == "graph_edit_distance":
                dist_fn = partial(nx.graph_edit_distance, 
                             node_match=node_match_on_pos, 
                             edge_match=edge_match_on_dep)
            elif self.config['similarity_type'] == "tree_edit_distance":
                dist_fn = zss_tree_edit_distance
                
            z = np.array([dist_fn(q_feat, f) for f in c_feat])

            # convert distance to similarity
            z = z - z.mean()
            z = 1 / (1+np.e**z)

        else:
            z = np.array([cos_sim(q_feat, f).item() for f in c_feat])

            # strongly penalize for any differences to make Z more intuitive
            z **= 200

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
    print_div_metric(DependencyDiversity, lo_div, hi_div)

    # similarities
    print("similarities")
    print_sim_metric(DependencyDiversity, lo_div, hi_div)

    # rank similarities
    print("rankings")
    print_ranking(DependencyDiversity, ["burn big planets"], lo_div + hi_div)

    # (textdiv) ~\GitHub\TextDiversity\src>python -m textdiversity.text_diversities.syntactic
