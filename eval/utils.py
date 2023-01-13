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
    def __init__(self, dataset_name, dataset):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.num_classes = len(self.dataset.features['label'].names)
        self.api = HfApi()
        self.pipe = None
        self.device = 0 if torch.cuda.is_available() else -1
        self.model_filter = ModelFilter(
            task="text-classification",
            library="pytorch",
            # model_name=dataset_name,
            trained_dataset=self.dataset_name)
        self.find_model_for_dataset()

    def find_model_for_dataset(self):
        model_id = next(iter(self.api.list_models(filter=self.model_filter)))
        if model_id:
            model_id = getattr(model_id, 'modelId')
            print('Using ' + model_id + ' to support cleanlab datalabel issues.')
            self.pipe = pipeline("text-classification", 
                                 model=model_id, 
                                 device=self.device, 
                                 top_k=None)

    def extract_prediction_probabilities(self):
        output = self.pipe(self.dataset['text'])
        return np.stack([vectorize(o) for o in output])

    def clean_dataset(self):
        if self.pipe is None:
            return self.dataset

        pred_probs = self.extract_prediction_probabilities()
        suss_idx = find_label_issues(
            labels=self.dataset['label'],
            pred_probs=pred_probs,  
            return_indices_ranked_by='self_confidence',
            min_examples_per_class=(self.dataset_len // self.num_classes) - 1
        )
        idx_to_keep = [i for i in range(len(self.dataset)) if i not in suss_idx]
        return self.dataset.select(idx_to_keep)


if __name__ == "__main__":
    from datasets import load_dataset

    dataset_name = "snips_built_in_intents"
    dataset = load_dataset(dataset_name)['train']

    print("Using cleanlab to cleanup dataset...")
    print(f"Original dataset length: {len(dataset)}")
    filter = CleanLabFilter(dataset_name, dataset)
    dataset = filter.clean_dataset()
    print(f"Filtered dataset length: {len(dataset)}")