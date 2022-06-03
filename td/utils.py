import os
import requests
import re
from tqdm import tqdm
import zipfile
import numpy as np
import torch
from spacy.lang.en import English

# consts
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
METRICS_DATA_DIR = os.path.join(DATA_DIR, 'with_metrics')
RESULTS_DIR = 'results'
EXPERIMENTS_DIR = os.path.join(DATA_DIR, 'experiments')
LABEL_VAL_FIELD = 'label_value'
LABEL_NAME_FIELD = 'label_name'
LABEL_PREFIX = 'label_'
METRIC_FIELD_PREFIX = 'metric_'

def dict_print(d, indent=0):
    # code from https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            dict_print(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def parse_path_list(path_str, default_path, file_extension='.csv'):
    csv_list = []
    input_split = [default_path] if path_str == '' else path_str.split(',')

    for path in input_split:
        if os.path.isfile(path) and path.endswith(file_extension):
            csv_list.append(path)
        elif os.path.isdir(path):
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    sub_path = os.path.join(subdir, file)
                    if os.path.isfile(sub_path) and sub_path.endswith(file_extension):
                        csv_list.append(sub_path)
        else:
            raise FileNotFoundError('[{}] not exists.'.format(path))

    return csv_list

def CamleCase2snake_case(string):
    # code from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()

def represents_int(s):
    # code from https://stackoverflow.com/questions/1265665/
    # how-can-i-check-if-a-string-represents-an-int-without-using-try-except
    try:
        int(s)
        return True
    except ValueError:
        return False

def lines_to_ngrams(lines, n=3):
    ngrams = []
    for s in lines:
        words = [e for e in s.replace('.','').replace('\n','').split(' ') if e != '']
        ngrams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    return ngrams

def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    # code from https://stackoverflow.com/questions/12734517/json-dumping-a-dict-throws-typeerror-keys-must-be-a-string
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            # delete old key
            del d[key]
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise

    return d

def download_and_place_data():

    if not os.path.exists(DATA_DIR):

        url = 'http://diversity-eval.s3-us-west-2.amazonaws.com/data.zip'
        target_zip = 'data.zip'
        response = requests.get(url, stream=True)

        # download
        print('Downloading data from [{}]...'.format(url))
        with open(target_zip, "wb") as handle:
            for data in tqdm(response.iter_content(), unit='B', unit_scale=True, unit_divisor=1024):
                handle.write(data)

        # place
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(target_zip)

def optimal_classification_accuracy(group_1, group_2):
    """
    find optimal classification accuracy in 1d feature space by exhaustively checking all separators.
    :param group_1: list of 1d data points
    :param group_2: list of 1d data points
    :return: optimal classification accuracy (ocr), and classification threshold (th)
    """
    accuracy_list = []
    th_list = []
    all_samples = group_1 + group_2
    for separator in all_samples:
        group_1_left = sum([v <= separator + 1e-5 for v in group_1])
        group_2_right = sum([v > separator + 1e-5 for v in group_2])
        acc = (group_1_left + group_2_right) / len(all_samples)
        th_list.append(separator)
        accuracy_list.append(acc if acc > 0.5 else 1 - acc)

    best_separator_idx = np.argmax(accuracy_list)
    oca = accuracy_list[best_separator_idx]
    th = th_list[best_separator_idx]

    return oca, th

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

if __name__ == '__main__':
    pass