import os
import requests
import re
from tqdm import tqdm
import zipfile
import numpy as np
import torch
from spacy.lang.en import English

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def merge_bpe(tok, boe, chars="##"):
    new_tok = []
    new_boe = []

    emb = []
    append = ""
    for t, e in zip(tok[::-1], boe[::-1]):
        t += append
        emb.append(e)
        if t.startswith(chars):
            append = t.replace(chars, "")
        else:
            append = ""
            new_tok.append(t)
            new_boe.append(np.stack(emb).mean(axis=0))
            emb = []  
    new_tok = np.array(new_tok)[::-1]
    new_boe = np.array(new_boe)[::-1]
    
    return new_tok, new_boe

def find_max_list(lists):
    list_len = [len(l) for l in lists]
    return max(list_len)

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def compute_pairwise(inputs, fn, diagonal_val=0, verbose=True):
    num_inputs = len(inputs)

    Z = np.eye(num_inputs)
    iu = np.triu_indices(num_inputs, k=1)
    il = (iu[1], iu[0])

    outer_loop = range(num_inputs)
    if verbose:
        outer_loop = tqdm(outer_loop)

    for i in outer_loop:
        for j in range(1, num_inputs - i):
            d = fn(inputs[i], inputs[i + j])
            Z[i][i + j] = d     
    Z[il] = Z[iu]

    if diagonal_val is not None:
        np.fill_diagonal(Z, diagonal_val)
        
    return Z

def clean_text(texts):
    texts = [text.replace("<br />", "") for text in texts]
    texts = [text.replace("...", ". ") for text in texts]
    texts = [text.replace("..", ". ") for text in texts]
    texts = [text.replace(".", ". ") for text in texts]
    texts = [text.replace("!", "! ") for text in texts]
    texts = [text.replace("?", "? ") for text in texts]
    texts = [text.strip() for text in texts]
    return texts

def split_sentences(texts):
    sentencizer = English()
    sentencizer.add_pipe("sentencizer")

    sentences = []
    for text in texts:
        sents = list(sentencizer(text).sents)
        sents = [s.text.strip() for s in sents if s.text.strip()]
        sentences.extend(sents)
      
    return sentences

def print_div_metric(metric_class, lo_div, hi_div):

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

def print_sim_metric(metric_class, lo_div, hi_div):

    div_metric = metric_class()
    sim_hi = div_metric.similarity(lo_div)
    sim_lo = div_metric.similarity(hi_div)

    print('metric:{0} | lo_div: {1:0.3f} | hi_div: {2:0.3f}'
        .format(
            metric_class.__name__,
            sim_lo, 
            sim_hi))

if __name__ == '__main__':
    pass