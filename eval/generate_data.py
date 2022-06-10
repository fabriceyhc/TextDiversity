import os
import argparse

import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk

from paraphrasers import (
    TextDiversityParaphraser,
    DiPSParaphraser,
    SowReapParaphraser,
    QCPGParaphraser
)

# argparse

parser = argparse.ArgumentParser(description='TextDiversity Paraphrase Dataset Generator')

parser.add_argument('--dataset-config', nargs='+', default=['paws', 'labeled_final'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--dataset-keys', nargs='+', default=['sentence1', 'sentence2'],
                    type=str, help='dataset info needed for load_dataset.')
parser.add_argument('--data-dir', type=str, default="./prepared_datasets/",
                    help='directory in which to save the augmented data')
parser.add_argument('--num-outputs', default=3, type=int, metavar='N',
                    help='augmentation multiplier - number of new inputs per 1 original')
parser.add_argument('--batch-size', default=10, type=int, metavar='N',
                    help='number of inputs to proccess per iteration')
parser.add_argument('--techniques', nargs='+', 
                    default=['dips', 'beam', 'diverse_beam', 'random'], # 'textdiv', 'qcpg', 'sowreap'
                    type=str, help='technique used to generate paraphrases')
# parser.add_argument('--num-train-per-class', nargs='+', default=[10, 200, 2500], type=int, 
#                     help='number of training examples per class')
# parser.add_argument('--num-valid-per-class', default=2000, type=int, metavar='N',
#                     help='number of validation examples per class')
parser.add_argument('--keep-original', default=True, action='store_true',
                    help='preserve original dataset in the updated one')
parser.add_argument('--force', default=False, action='store_true',
                    help='force the dataset creation even if one already exists')
parser.add_argument('--save-file', type=str, default='prepared_dataset_log.csv',
                    help='name for the csv file to save with results')

args = parser.parse_args()


class DatasetAugmenter:
    def __init__(self, technique='textdiv', num_outputs=5):
        self.technique = technique
        
        if self.technique == 'textdiv':
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

    def __call__(self, batch):
        if len(args.dataset_keys) == 1:
            new_text, new_targets = [], []
            for s1, t in zip(batch[args.dataset_keys[0]], batch['label']):
                new_text.extend(self.transform_fn(s1))
                new_targets.extend([t] * self.num_outputs)
            return {args.dataset_keys[0]: new_text, 
                    "label": new_targets}
        elif len(args.dataset_keys) == 2:  
            new_text1, new_text2, new_targets = [], [], []
            for s1, s2, t in zip(batch[args.dataset_keys[0]], 
                                 batch[args.dataset_keys[1]], 
                                 batch['label']):
                new_text1.extend(self.transform_fn(s1))
                new_text2.extend(self.transform_fn(s2))
                new_targets.extend([t] * self.num_outputs)
            return {args.dataset_keys[0]: new_text1, 
                    args.dataset_keys[1]: new_text2, 
                    "label": new_targets}
        else:
            raise ValueError("Unsupported number of dataset-keys. Should be either 1 or 2.")


# load dataset and process
dataset = load_dataset(*args.dataset_config, split='train') 
# ugly way of standardizing dataset column names
try:
    # for glue.qqp
    dataset = dataset.rename_column("question1", "sentence1")
    dataset = dataset.rename_column("question2", "sentence2")
except:
    pass
try:
    # for glue.sst2
    dataset = dataset.rename_column("sentence", "text")
except:
    pass

key_columns = ['text', 'sentence1', 'sentence2', 'label']
columns_to_remove = [c for c in dataset.column_names if c not in key_columns]
dataset = dataset.remove_columns(columns_to_remove)

original_dataset_len = len(dataset)
print('original_dataset_len:', original_dataset_len)

# generate updated datasets for each paraphrase technique
for technique in args.techniques:

    print(f'Running {technique}')

    save_name = "_".join(args.dataset_config) + "_" + technique
    save_path = os.path.join(args.data_dir, save_name)

    if os.path.exists(save_path) and not args.force:
        print(f"existing dataset found at {save_path}. skipping...")
        continue

    # load paraphrase augmentation technique
    augmenter = DatasetAugmenter(technique=technique, 
                                 num_outputs=args.num_outputs)

    # augment dataset
    updated_dataset = dataset.map(augmenter, 
                                  batched=True, 
                                  batch_size=args.batch_size)

    updated_dataset_len = len(updated_dataset)
    print('updated_dataset_len:', updated_dataset_len)

    # check for expected number of data points
    assert original_dataset_len * args.num_outputs == updated_dataset_len

    # check for new data actually being paraphrases (not duplicates)
    assert len(set(updated_dataset[-args.num_outputs:][args.dataset_keys[0]])) > 1 

    if args.keep_original:
        updated_dataset = concatenate_datasets([dataset, updated_dataset])

    # save updated dataset
    updated_dataset.save_to_disk(save_path)

    # print examples of updated dataset just to be sure they're as expected
    num_to_check = args.num_outputs * 3
    print('example:', updated_dataset[-num_to_check:])

    torch.cuda.empty_cache()