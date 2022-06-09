import os
import argparse

from datasets import load_dataset

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
parser.add_argument('--data-dir', type=str, default="./prepared_datasets/",
                    help='directory in which to save the augmented data')
parser.add_argument('--num-outputs', default=3, type=int, metavar='N',
                    help='augmentation multiplier - number of new inputs per 1 original')
parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                    help='number of inputs to proccess per iteration')
parser.add_argument('--techniques', nargs='+', 
                    default=['beam', 'diverse_beam', 'random',
                             'textdiv', 'dips', 'qcpg'], # sowreap
                    type=str, help='technique used to generate paraphrases')
# parser.add_argument('--num-train-per-class', nargs='+', default=[10, 200, 2500], type=int, 
#                     help='number of training examples per class')
# parser.add_argument('--num-valid-per-class', default=2000, type=int, metavar='N',
#                     help='number of validation examples per class')
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
        self.transform_fn.num_outputs = num_outputs

    def __call__(self, batch):
        new_sentence1, new_sentence2, new_targets = [], [], []
        for s1, s2, t in zip(batch['sentence1'], batch['sentence2'], batch['label']):
            new_sentence1.append(self.transform_fn(s1))
            new_sentence2.append(self.transform_fn(s2))
            new_targets.append([t] * self.num_outputs)
        return {"sentence1": new_sentence1, 
                "sentence2": new_sentence2, 
                "label": new_targets}


# load dataset and process
dataset = load_dataset(*args.dataset_config, split='train') 

for technique in args.techniques:

    print(f'Running {technique}')

    # load paraphrase augmentation technique
    augmenter = DatasetAugmenter(technique=technique, 
                                 num_outputs=args.num_outputs)

    # augment dataset
    updated_dataset = dataset.map(augmenter, 
                                  batched=True, 
                                  batch_size=args.batch_size)

    # save updated dataset
    save_name = "_".join(args.dataset_config) + "_" + technique
    save_path = os.path.join(args.data_dir, save_name)
    updated_dataset.save_to_disk(save_path)