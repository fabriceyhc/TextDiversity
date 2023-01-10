from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    EarlyStoppingCallback
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_callback import TrainerControl
from datasets import load_dataset, load_metric, load_from_disk
import os
import sys
import gc
import argparse
import time
import random
import shutil
import torch
import numpy as np
import pandas as pd

import evaluate
from nltk import ngrams

from paraphrasers import (
    TextDiversityParaphraser,
    DiPSParaphraser,
    SowReapParaphraser,
    QCPGParaphraser,
    QCPGPPlusPlusaraphraser
)

from textdiversity import (
    TokenSemantics,
    DocumentSemantics,
    PartOfSpeechSequence,
    Rhythmic,
    Phonemic,
    DependencyParse,
)

# aargparse

parser = argparse.ArgumentParser(description='TextDiversity Trainer')

parser.add_argument('--data-dir', type=str, default="./prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--num-runs', default=1, type=int, metavar='N',
                    help='number of times to repeat the evaluation')
parser.add_argument('--num-eval', default=5, type=int, metavar='N',
                    help='number of samples to draw from the dataset for evaluation')
parser.add_argument('--techniques', nargs='+', 
                    default=['orig', 'beam', 'diverse_beam', 'random', 'qcpg', 'qcpgpp', 'textdiv', 'dips'], # 'sowreap'
                    type=str, help='technique used to generate paraphrases')
parser.add_argument('--dataset-config', nargs='+', default=['paws', 'labeled_final'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-keys', nargs='+', default=['sentence1', 'sentence2'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--save-file', type=str, default='instrinsic_results.csv',
                    help='name for the csv file to save with results')
parser.add_argument('--use-cuda', default=False, action='store_true',
                    help='whether or not to use cuda')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() and args.use_cuda else torch.device('cpu')
print(f"device = {device}")

#############################################################
## Helpers ##################################################
#############################################################

def ngram_toks(corpus, n=1):
    ntoks = []
    for doc in corpus:
        ntok = list(ngrams(doc.split(), n))
        newtoks = [tok for tok in ntok]
        ntoks += newtoks
    return ntoks

def distinct_ngrams(corpus, n):
    if len(corpus):
        corpus = " ".join(corpus)
        toks = set(ngram_toks([corpus], n))
        score = len(toks)
        return score
    else:
        return 0.0

class Paraphraser:
    def __init__(self, technique='textdiv', num_outputs=5, use_cuda=True):
        self.technique = technique
        
        if self.technique == 'orig':
            self.transform_fn = lambda x: [x] * num_outputs
        elif self.technique == 'random':
            self.transform_fn = DiPSParaphraser(use_cuda=use_cuda, augmenter='random')
        elif self.technique == 'beam':
            self.transform_fn = DiPSParaphraser(use_cuda=use_cuda, augmenter='beam')
        elif self.technique == 'diverse_beam':
            self.transform_fn = DiPSParaphraser(use_cuda=use_cuda, augmenter='diverse_beam')
        elif self.technique == 'dips':
            self.transform_fn = DiPSParaphraser(use_cuda=use_cuda, augmenter='dips')
        elif self.technique == 'textdiv':
            self.transform_fn = TextDiversityParaphraser(use_cuda=use_cuda)
        elif self.technique == 'qcpgpp':
            self.transform_fn = QCPGPPlusPlusaraphraser(use_cuda=use_cuda, num_outputs=num_outputs)
        elif self.technique == 'qcpg':
            self.transform_fn = QCPGParaphraser(use_cuda=use_cuda, num_outputs=num_outputs)
        elif self.technique == 'sowreap':
            self.transform_fn = SowReapParaphraser()
        else:
            raise ValueError("must provide a valid paraphrase generation technique.")
        
        self.num_outputs = num_outputs
        self.transform_fn.num_outputs = self.num_outputs

    def __call__(self, text):
        return self.transform_fn(text)

class DiversityEvaluator:
    def __init__(self):
        self.tokdiv_fn = TokenSemantics({'use_cuda':args.use_cuda})
        self.docdiv_fn = DocumentSemantics({'use_cuda':args.use_cuda})
        self.posdiv_fn = PartOfSpeechSequence({'use_cuda':args.use_cuda})
        self.rhydiv_fn = Rhythmic({'use_cuda':args.use_cuda})
        self.phodiv_fn = Phonemic({'use_cuda':args.use_cuda})
        self.depdiv_fn = DependencyParse({'use_cuda':args.use_cuda})

    def __call__(self, corpus):

        results = {}

        # distinct n-grams
        for n in [1,2,3,4]:
            metric_name = f'distinct_{n}-grams'
            print(f'Working on {metric_name}...')
            results[metric_name] = distinct_ngrams(corpus, n)

        # textdiversity fns
        fns = [self.tokdiv_fn, self.docdiv_fn, 
               self.posdiv_fn, self.rhydiv_fn,
               self.phodiv_fn, self.depdiv_fn]
        for fn in fns:
            metric_name = fn.__class__.__name__
            print(f'Working on {metric_name}...')
            results[metric_name] = fn(corpus)

        return results

class FidelityEvaluator:
    def __init__(self):
        self.sacrebleu = evaluate.load("sacrebleu")
        self.meteor = evaluate.load('meteor')
        self.bleurt = evaluate.load('bleurt')

    def __call__(self, predictions, references):

        results = {}

        # huggingface evaluate fns
        fns = [self.sacrebleu, self.meteor, self.bleurt]
        for fn in fns:
            metric_name = fn.__class__.__name__
            print(f'Working on {metric_name}...')
            scoring = fn.compute(predictions=predictions,
                                 references=references)
            if metric_name == 'Meteor':
                score = scoring['meteor']
            elif metric_name == 'BLEURT':
                score = np.mean(scoring['scores'])
            elif metric_name == 'Sacrebleu':
                score = scoring['score']
            results[metric_name] = score
        return results

#############################################################
## Main Loop Functionality ##################################
#############################################################

div_evaluator = DiversityEvaluator()
fid_evaluator = FidelityEvaluator()

# load dataset and process
if 'trec' in args.dataset_config:
    train_dataset = load_dataset(args.dataset_config[0], split='train') 
    if 'coarse_label' in args.dataset_config:
        train_dataset = train_dataset['train'].remove_columns("fine_label")
        train_dataset = train_dataset.rename_column("coarse_label", "label")
    elif 'fine_label' in args.dataset_config:
        train_dataset = train_dataset['train'].remove_columns("coarse_label")
        train_dataset = train_dataset.rename_column("fine_label", "label")
else:
    train_dataset = load_dataset(*args.dataset_config, split='train') 

results = []
for run_num in range(args.num_runs):

    # select random data to analyze
    idx = random.randint(0, len(train_dataset))
    row_to_paraphrase = train_dataset.select([idx])[0]

    if len(args.dataset_keys) == 1:
        sentence1_key, sentence2_key = args.dataset_keys[0], None
        text_to_paraphrase = row_to_paraphrase[sentence1_key]
    else:
        # if not 1 then assume 2 keys
        sentence1_key, sentence2_key = args.dataset_keys
        text_to_paraphrase = row_to_paraphrase[sentence1_key] + ' ' + row_to_paraphrase[sentence2_key]

    for technique in args.techniques:

        print(f'run_num: {run_num}, technique: {technique}')

        # paraphrase as we go 
        paraphraser = Paraphraser(technique=technique, num_outputs=args.num_eval, use_cuda=args.use_cuda)

        #############################################################
        ## Intrinsic Evaluation #####################################
        #############################################################

        start_time = time.time()
        print('text_to_paraphrase:', text_to_paraphrase)
        paraphrases = paraphraser(text_to_paraphrase)
        print('paraphrases:', paraphrases)
        div_eval_results = div_evaluator(paraphrases)
        fid_eval_results = fid_evaluator(paraphrases, [text_to_paraphrase] * args.num_eval)
        out = {**div_eval_results, **fid_eval_results}
        run_time = time.time() - start_time

        out['text_to_paraphrase'] = text_to_paraphrase
        out['paraphrases'] = paraphrases
        out['run_num'] = run_num
        out['technique'] = technique
        out['dataset_config'] = args.dataset_config
        out['run_time'] = run_time
        print(f'Performance: {out}')

        results.append(out)

        # save results
        df = pd.DataFrame(results)
        df.to_csv(args.save_file)

        # clear out memory for next round
        gc.collect()
        torch.cuda.empty_cache()