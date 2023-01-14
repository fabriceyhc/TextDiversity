import torch
import numpy as np
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter
from cleanlab.filter import find_label_issues

def vectorize(output):
    sorted_output = sorted(output, key=lambda d: d['label']) 
    probs = np.array([d['score'] for d in sorted_output])
    return probs

class CleanLabFilter:
    def __init__(self):
        self.api = HfApi()
        self.pipe = None
        self.device = 0 if torch.cuda.is_available() else -1

    def find_model_for_dataset(self, dataset_name):
        
        model_filter = ModelFilter(
            task="text-classification",
            library="pytorch",
            # model_name=dataset_name,
            trained_dataset=dataset_name)

        model_id = next(iter(self.api.list_models(filter=model_filter)))

        if model_id:
            model_id = getattr(model_id, 'modelId')
            print('Using ' + model_id + ' to support cleanlab datalabel issues.')
            self.pipe = pipeline("text-classification", 
                                 model=model_id, 
                                 device=self.device, 
                                 top_k=None)

    def extract_prediction_probabilities(self, dataset):
        output = self.pipe(dataset['text'])
        return np.stack([vectorize(o) for o in output])

    def find_num_to_remove_per_class(self, dataset, frac_to_remove=0.1):
        classes = dataset.features['label'].names
        num_classes = len(classes)
        print(num_classes)

        num_per_class = []
        for i in range(num_classes):
            class_partition = dataset.filter(lambda row: row["label"] == i)
            num_per_class.append(len(class_partition))
        num_to_remove_per_class = [int(frac_to_remove * num) for num in num_per_class]
        return num_to_remove_per_class

    def clean_dataset(self, dataset):
        if self.pipe is None:
            return dataset
    
        dataset_len = len(dataset)
        num_classes = len(dataset.features['label'].names)

        pred_probs = self.extract_prediction_probabilities(dataset)
        print(f"pred_probs.shape ({pred_probs.shape})")

        num_to_remove_per_class = self.find_num_to_remove_per_class(dataset)
        print(f"num_to_remove_per_class ({num_to_remove_per_class})")

        num_to_add = pred_probs.shape[-1] - len(num_to_remove_per_class)
        print(f"num_to_add: {num_to_add}")
        for i in range(num_to_add):
            num_to_remove_per_class.append(0)

        suss_idx = find_label_issues(
            labels=dataset['label'],
            pred_probs=pred_probs,  
            return_indices_ranked_by='self_confidence',
            num_to_remove_per_class = num_to_remove_per_class
        )
        print(f"suss_idx.len ({len(suss_idx)})")
        idx_to_keep = [i for i in range(len(dataset)) if i not in suss_idx]
        print(f"idx_to_keep.len ({len(idx_to_keep)})")
        return dataset.select(idx_to_keep)


if __name__ == "__main__":
    from datasets import load_dataset

    dataset_name = "snips_built_in_intents"
    dataset = load_dataset(dataset_name)['train']

    cl_filter = CleanLabFilter()
    cl_filter.find_model_for_dataset(dataset_name)

    for i in range(3):
        print("Using cleanlab to cleanup dataset...")
        print(f"Original dataset length: {len(dataset)}")
        dataset = cl_filter.clean_dataset(dataset)
        print(f"Filtered dataset length: {len(dataset)}")