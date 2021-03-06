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
import argparse
import time
import random
import shutil
import torch
import pandas as pd
from torch.utils.data import DataLoader

# aargparse

parser = argparse.ArgumentParser(description='TextDiversity Trainer')

parser.add_argument('--data-dir', type=str, default="./prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--save-dir', type=str, default="./pretrained/",
                    help='path to data folders')
parser.add_argument('--num_epoch', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--eval-batch-size', default=16, type=int, metavar='N',
                    help='eval batchsize')
parser.add_argument('--gpus', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# parser.add_argument('--transformers_cache', default="../../data1/fabricehc/.cache", type=str,
#                     help='location for for TRANSFORMERS_CACHE')
# parser.add_argument('--datasets_cache', default="../../data1/fabricehc/.cache", type=str,
#                     help='location for for HF_DATASETS_CACHE')
parser.add_argument('--num_runs', default=1, type=int, metavar='N',
                    help='number of times to repeat the training')
parser.add_argument('--techniques', nargs='+', 
                    default=['orig', 'dips', 'beam', 'diverse_beam', 'random'], #  'textdiv', 'qcpg', 'sowreap'
                    type=str, help='technique used to generate paraphrases')
parser.add_argument('--dataset-config', nargs='+', default=['paws', 'labeled_final'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-keys', nargs='+', default=['sentence1', 'sentence2'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--models', nargs='+',  default=['roberta-base'], 
                    type=str, help='pretrained huggingface models to train')
parser.add_argument('--save-file', type=str, default='train_results.csv',
                    help='name for the csv file to save with results')

args = parser.parse_args()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#############################################################
## Helper Functions #########################################
#############################################################

def tokenize_fn(batch):
    if sentence2_key is None:
        return tokenizer(batch[sentence1_key], padding=True, truncation=True, max_length=250)
    return tokenizer(batch[sentence1_key], batch[sentence2_key], padding=True, truncation=True, max_length=250)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if len(labels.shape) > 1: 
        acc = acc_at_k(labels, predictions, k=2)
        return { 'accuracy': acc }        
    else:
        acc = accuracy_score(labels, predictions.argmax(-1))
        return { 'accuracy': acc } 

#############################################################
## Main Loop Functionality ##################################
#############################################################

run_args = []
for run_num in range(args.num_runs):
    for technique in args.techniques:
        for MODEL_NAME in args.models:
            run_args.append({
                "run_num":run_num,
                "technique":technique,
                "MODEL_NAME":MODEL_NAME
            })

print(run_args)

results = []
save_file = args.save_file  
if os.path.exists(save_file):
    results.extend(pd.read_csv(save_file).to_dict("records"))
    start_position = len(results)
else:
    start_position = 0

print('starting at position {}'.format(start_position))
for run_arg in run_args[start_position:]:

    #############################################################
    ## Initializations ##########################################
    #############################################################
    run_num = run_arg['run_num']
    technique = run_arg['technique']
    MODEL_NAME = run_arg['MODEL_NAME']

    print(pd.DataFrame([run_arg]))

    if len(args.dataset_keys) == 1:
        sentence1_key, sentence2_key = args.dataset_keys[0], None
    else:
        # if not 1 then assume 2 keys
        sentence1_key, sentence2_key = args.dataset_keys

    #############################################################
    ## Dataset Preparation ######################################
    #############################################################

    if technique == "orig":
        train_dataset = load_dataset(*args.dataset_config, split='train').shuffle()
    else:
        save_name = "_".join(args.dataset_config) + "_" + technique
        save_path = os.path.join(args.data_dir, save_name)
        train_dataset = load_from_disk(save_path).shuffle()

    if "snips_built_in_intents" in args.dataset_config:
        # special handling since snips has no val / test split
        train_testvalid = train_dataset.train_test_split(test_size=0.1)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        train_dataset = train_testvalid['train'].shuffle()
        eval_dataset  = test_valid['test']
        test_dataset  = test_valid['train']
    elif 'sst2' in args.dataset_config:
        # special handling since sst2 has no test labels
        eval_dataset  = load_dataset(*args.dataset_config, split='validation')
        test_valid = eval_dataset.train_test_split(test_size=0.5)
        eval_dataset  = test_valid['test']
        test_dataset  = test_valid['train']
    elif 'banking77' in args.dataset_config:
        # special handling since banking77 has no validation split
        test_dataset  = load_dataset(*args.dataset_config, split='test')
        test_valid = test_dataset.train_test_split(test_size=0.5)
        eval_dataset  = test_valid['test']
        test_dataset  = test_valid['train']
    else:
        eval_dataset  = load_dataset(*args.dataset_config, split='validation')
        test_dataset  = load_dataset(*args.dataset_config, split='test')

    num_classes = train_dataset.features['label'].num_classes

    print('Length of train_dataset:', len(train_dataset))
    print('Length of eval_dataset:', len(eval_dataset))
    print('Number of classes:', num_classes)

    #############################################################
    ## Model + Tokenize #########################################
    #############################################################
    checkpoint = args.save_dir + MODEL_NAME + '-' + "_".join(args.dataset_config) + "_" + technique
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes).to(device)

    # tokenize datasets
    train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=1000)
    train_dataset = train_dataset.rename_column("label", "labels") 
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, batch_size=len(eval_dataset))
    eval_dataset = eval_dataset.rename_column("label", "labels") 
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_dataset = test_dataset.map(tokenize_fn, batched=True, batch_size=len(test_dataset))
    test_dataset = test_dataset.rename_column("label", "labels") 
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    #############################################################
    ## Callbacks ################################################
    #############################################################

    callbacks = []

    tmcb = None   
    escb = EarlyStoppingCallback(
        early_stopping_patience=5
    )
    callbacks.append(escb)

    #############################################################
    ## Training  ################################################
    #############################################################

    train_batch_size = args.train_batch_size    
    eval_batch_size = args.eval_batch_size  
    num_epoch = args.num_epoch  
    gradient_accumulation_steps = 1
    max_steps = int((len(train_dataset) * num_epoch / gradient_accumulation_steps) / train_batch_size)
    logging_steps = max_steps // num_epoch

    training_args = TrainingArguments(
        output_dir=checkpoint,
        overwrite_output_dir=True,
        max_steps=max_steps,
        save_steps=logging_steps,
        save_total_limit=1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        warmup_steps=int(max_steps / 10),
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=logging_steps,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        evaluation_strategy="steps",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,                  
        train_dataset=train_dataset,         
        eval_dataset=eval_dataset,
        callbacks=callbacks
    )

    start_time = time.time()
    trainer.train()
    run_time = time.time() - start_time

    # test with ORIG data
    trainer.eval_dataset = test_dataset

    out = trainer.evaluate()

    out['run_name'] = checkpoint
    out['model_name'] = MODEL_NAME
    out['run_num'] = run_num
    out['technique'] = technique
    out['dataset_config'] = args.dataset_config
    out['run_time'] = run_time
    print('Performance of {}\n{}'.format(checkpoint, out))

    results.append(out)

    # save results
    df = pd.DataFrame(results)
    df.to_csv(save_file)
