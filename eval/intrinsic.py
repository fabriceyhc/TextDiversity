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
import pandas as pd

import evaluate
from nltk import ngrams

from paraphrasers import (
    TextDiversityParaphraser,
    DiPSParaphraser,
    SowReapParaphraser,
    QCPGParaphraser
)

from textdiversity import (
    TokenSemanticDiversity,
    DocumentSemanticDiversity,
    POSSequenceDiversity,
    RhythmicDiversity,
    PhonemicDiversity,
    DependencyDiversity,
)

# aargparse

parser = argparse.ArgumentParser(description='TextDiversity Trainer')

parser.add_argument('--data-dir', type=str, default="./prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--num-runs', default=1, type=int, metavar='N',
                    help='number of times to repeat the training')
parser.add_argument('--num-eval', default=5, type=int, metavar='N',
                    help='number of samples to draw from the dataset for evaluation')
parser.add_argument('--techniques', nargs='+', 
                    default=['orig', 'dips', 'beam', 'diverse_beam', 'random','textdiv'], # , 'qcpg', 'sowreap'
                    type=str, help='technique used to generate paraphrases')
parser.add_argument('--dataset-config', nargs='+', default=['paws', 'labeled_final'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-keys', nargs='+', default=['sentence1', 'sentence2'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--save-file', type=str, default='instrinsic_results.csv',
                    help='name for the csv file to save with results')

args = parser.parse_args()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    def __init__(self, technique='textdiv', num_outputs=5):
        self.technique = technique
        
        if self.technique == 'orig':
            self.transform_fn = lambda x: [x] * num_outputs
        elif self.technique == 'textdiv':
            self.transform_fn = TextDiversityParaphraser()
        elif self.technique == 'dips':
            self.transform_fn = DiPSParaphraser(augmenter='dips')
        elif self.technique == 'sowreap':
            self.transform_fn = SowReapParaphraser()
        elif self.technique == 'qcpg':
            self.transform_fn = QCPGParaphraser()
        elif self.technique == 'beam':
            self.transform_fn = DiPSParaphraser(augmenter='beam')
        elif self.technique == 'diverse_beam':
            self.transform_fn = DiPSParaphraser(augmenter='diverse_beam')
        elif self.technique == 'random':
            self.transform_fn = DiPSParaphraser(augmenter='random')
        else:
            raise ValueError("must provide a valid paraphrase generation technique.")
        
        self.num_outputs = num_outputs
        self.transform_fn.num_outputs = self.num_outputs

    def __call__(self, text):
        return self.transform_fn(text)

class DiversityEvaluator:
    def __init__(self):
        self.tokdiv_fn = TokenSemanticDiversity()
        self.docdiv_fn = DocumentSemanticDiversity()
        self.posdiv_fn = POSSequenceDiversity()
        self.rhydiv_fn = RhythmicDiversity()
        self.phodiv_fn = PhonemicDiversity()
        self.depdiv_fn = DependencyDiversity()

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
        self.bleurt = evaluate.load('bleurt', 'bleurt-large-512')

    def __call__(self, predictions, references):

        results = {}

        # huggingface evaluate fns
        fns = [self.sacrebleu, self.meteor, self.bleurt]
        for fn in fns:
            metric_name = fn.__class__.__name__
            print(f'Working on {metric_name}...')
            results = fn.compute(predictions=predictions,
                                 references=references)
            if metric_name == 'Meteor':
                score = results['meteor']
            elif metric_name == 'BLEURT':
                score = np.mean(results['scores'])
            elif metric_name == 'Sacrebleu':
                score = results['score']
            results[metric_name] = score
        return results

#############################################################
## Main Loop Functionality ##################################
#############################################################

div_evaluator = DiversityEvaluator()
fid_evaluator = FidelityEvaluator()

results = []
for run_num in range(args.num_runs):

    train_dataset = load_dataset(*args.dataset_config, split='train')
    row_to_paraphrase = train_dataset.shuffle()[0]

    if len(args.dataset_keys) == 1:
        sentence1_key, sentence2_key = args.dataset_keys[0], None
        text_to_paraphrase = row_to_paraphrase[sentence1_key]
    else:
        # if not 1 then assume 2 keys
        sentence1_key, sentence2_key = args.dataset_keys
        text_to_paraphrase = row_to_paraphrase[sentence1_key] + row_to_paraphrase[sentence2_key]

    for technique in args.techniques:

        print(f'run_num: {run_num}, technique: {technique}')

        # paraphrase as we go 
        paraphraser = Paraphraser(technique=technique, num_outputs=args.num_eval)

        #############################################################
        ## Intrinsic Evaluation #####################################
        #############################################################

        start_time = time.time()
        print('text_to_paraphrase:', text_to_paraphrase)
        paraphrases = paraphraser(text_to_paraphrase)
        print('paraphrases:', paraphrases)
        div_eval_results = div_evaluator(paraphrases)
        fid_eval_results = fid_evaluator(paraphrases, [text_to_paraphrase] * args.num_eval)
        out = div_eval_results | fid_eval_results
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