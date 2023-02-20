from datasets import (
    load_dataset,
    load_from_disk
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate

import os
import argparse
import time
import torch
import numpy as np
import pandas as pd
import random

from filters import CleanLabFilter, balance_dataset
from utils import *

random.seed(130)
torch.use_deterministic_algorithms(False)

# aargparse

parser = argparse.ArgumentParser(description='TextDiversity Trainer')

parser.add_argument('--techniques', nargs='+', 
                    default=['orig', 'beam', 'diverse_beam', 'random', 'qcpg', 'qcpgpp', 'textdiv', 'dips'], #  'textdiv', 'qcpg', 'sowreap'
                    type=str, help='technique used to generate paraphrases')
parser.add_argument('--dataset-config', nargs='+', default=['paws', 'labeled_final'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-text-keys', nargs='+', default=['text'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-label-keys', nargs='+', default=['label'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--models', nargs='+',  default=['prajjwal1/bert-tiny', 'bert-base-uncased'], 
                    type=str, help='pretrained huggingface models to train')
parser.add_argument('--data-dir', type=str, default="./prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--save-dir', type=str, default="./pretrained/",
                    help='path to data folders')
parser.add_argument('--num_epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--logging_steps_per_epoch', default=10, type=int, metavar='N',
                    help='number of times to run validation + log per epoch')
parser.add_argument('--early_stopping_patience', default=10, type=int, metavar='N',
                    help='number of times the validation metric can be lower before stopping training')
parser.add_argument('--gradient_accumulation_steps', default=1, type=int, metavar='N',
                    help='number of steps before updating the model weights')
parser.add_argument('--train-batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--eval-batch-size', default=16, type=int, metavar='N',
                    help='eval batchsize') 
parser.add_argument('--gpus', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_runs', default=3, type=int, metavar='N',
                    help='number of times to repeat the training')
parser.add_argument('--num-train-per-class', nargs='+', default=[10, "all"], type=int, 
                    help='number of training examples per class')
parser.add_argument('--cleanlab-filter',  nargs='+', default=[False],
                    help='filter out inputs with potential label errors')
parser.add_argument('--save-file', type=str, default='./results/train/techniques.csv',
                    help='name for the csv file to save with results')

args = parser.parse_args()
# print(args)

#############################################################
## Main Loop Functionality ##################################
#############################################################

def train(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(f"device={device}")

    run_args = []
    for run_num in range(args.num_runs):
        for model in args.models:
            for use_cleanlab in args.cleanlab_filter:
                for technique in args.techniques:
                    for num_train_per_class in args.num_train_per_class:
                        run_args.append({
                            "run_num":run_num,
                            "technique":technique,
                            "num_train_per_class":num_train_per_class,
                            "model":model,
                            "use_cleanlab":use_cleanlab,
                        })

    print(run_args)

    results = []
    save_file = args.save_file  
    if os.path.exists(save_file):
        results.extend(pd.read_csv(save_file).to_dict("records"))
        start_position = len(results)
    else:
        start_position = 0

    cl_filter = CleanLabFilter()

    print('starting at position {}'.format(start_position))
    for run_arg in run_args[start_position:]:

        #############################################################
        ## Initializations ##########################################
        #############################################################
        run_num = run_arg['run_num']
        technique = run_arg['technique']
        model_checkpoint = run_arg['model']

        print(pd.DataFrame([run_arg]))

        if len(args.dataset_text_keys) == 1:
            text1_key, text2_key = args.dataset_text_keys[0], None
        else:
            # if not 1 then assume 2 keys
            text1_key, text2_key = args.dataset_text_keys

        label_key = args.dataset_label_keys[0]

        #############################################################
        ## Dataset Preparation ######################################
        #############################################################

        raw_datasets = load_dataset(*args.dataset_config)
        if technique != "orig":
            save_name = "_".join(args.dataset_config) + "_" + technique
            save_path = os.path.join(args.data_dir, save_name)
            train_dataset = load_from_disk(save_path)
            raw_datasets["train"] = train_dataset

        raw_datasets = prepare_splits(raw_datasets, label_key=label_key)
        raw_datasets = raw_datasets.shuffle(seed=run_num)

        # remove suspicious labels using cleanlab
        if use_cleanlab:
            print("Using cleanlab to cleanup dataset...")
            print(f"Original dataset length: {len(train_dataset)}")
            cl_filter.find_model_for_dataset(args.dataset_config[0])
            train_dataset = cl_filter.clean_dataset(train_dataset)
            print(f"Filtered dataset length: {len(train_dataset)}")

        num_labels = len(raw_datasets["train"].features["label"].names)

        print(raw_datasets)
        print('Number of classes:', num_labels)

        #############################################################
        ## Model + Tokenize #########################################
        #############################################################
        
        checkpoint = args.save_dir + model_checkpoint + '-' + "_".join(args.dataset_config) + "_" + technique
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)

        # tokenize datasets
        def preprocess_function(batch):
            if text2_key is None:
                return tokenizer(batch[text1_key], padding=True, truncation=True)
            return tokenizer(batch[text1_key], batch[text2_key], padding=True, truncation=True)

        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

        #############################################################
        ## Metrics ##################################################
        #############################################################

        metrics = evaluate.combine([
            evaluate.load('accuracy'), 
            ConfiguredMetric(evaluate.load('precision'), average='weighted'),
            ConfiguredMetric(evaluate.load('recall'), average='weighted'),
            ConfiguredMetric(evaluate.load('f1'), average='weighted'),
        ])

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            return metrics.compute(predictions=preds, references = labels)

        #############################################################
        ## Callbacks ################################################
        #############################################################

        callbacks = []
        escb = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        callbacks.append(escb)

        #############################################################
        ## Training  ################################################
        #############################################################

        max_steps = (len(tokenized_datasets["train"]) * args.num_epochs // args.gradient_accumulation_steps) // args.train_batch_size
        logging_steps = (max_steps // args.num_epochs) // args.logging_steps_per_epoch

        training_args = TrainingArguments(
            output_dir=checkpoint,
            overwrite_output_dir=True,
            max_steps=max_steps,
            save_steps=logging_steps,
            save_total_limit=1,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps, 
            warmup_steps=logging_steps,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=logging_steps,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            evaluation_strategy="steps",
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model, 
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,                  
            train_dataset=tokenized_datasets["train"],         
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            callbacks=callbacks
        )

        start_time = time.time()
        trainer.train()
        run_time = time.time() - start_time

        # test with ORIG data
        out = trainer.evaluate(tokenized_datasets["test"])
        out.update({
            "run_num":          run_num,
            "technique":        technique,
            "model_checkpoint": model_checkpoint,
            "checkpoint":       checkpoint,
            "dataset_config":   args.dataset_config,
            "train_size":       len(tokenized_datasets["train"]),
            "valid_size":       len(tokenized_datasets["validation"]),
            "test_size":        len(tokenized_datasets["test"]),
            "run_time":         run_time,
        })
        print('Performance of {}\n{}'.format(checkpoint, out))

        results.append(out)

        # save results
        df = pd.DataFrame(results)
        df.to_csv(save_file, index=False)


if __name__ == "__main__":
    train(args)