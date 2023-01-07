from collections import defaultdict
from transformers import pipeline
import numpy as np
import torch

# GENERATION APPROACHES 

class QCPGPPlusPlusaraphraser:
    def __init__(self, num_outputs=1, use_cuda=True, sample_controls_from_ranges=True):
        self.device = 0 if use_cuda and torch.cuda.is_available() else -1
        self.pipe = pipeline('text2text-generation', 
                             model='madhavsankar/qcpg-parabk2-sbert-lr1e-4',
                             do_sample=True,
                             num_return_sequences=num_outputs,
                             device=self.device)
        self.diversities = ["semantic_sim", "syntactic_div", "morphological_div", "phonological_div", "lexical_div"]
        self.sample_controls_from_ranges = sample_controls_from_ranges
        self.control_options = find_control_options(self.pipe.tokenizer)
        self.control_ranges = {
            "semantic_sim": (0.8, 0.9),
            "syntactic_div": (0.7, 0.9),
            "morphological_div": (0.7, 0.9),
            "phonological_div": (0.7, 0.9),
            "lexical_div": (0.7, 0.7),
        }

    def __call__(self, text, semantic=0.8, syntactic=0.5, morphologic=0.5, phonologic=0.5, lexical=0.5, **kwargs):
        if self.sample_controls_from_ranges:
            control = {div: np.random.uniform(lo, hi) for div, (lo, hi) in self.control_ranges.items()}
        else:
            control = {div: val for div, val in zip(self.diversities, [semantic, syntactic, morphologic, phonologic, lexical])}
        control = [f'COND_{div.upper()}_{nearest_value(val * 100, self.control_options[div])}' for div, val in control.items()]
        text = ' '.join(control) + ' ' + text if isinstance(text, str) else [' '.join(control) for t in text]
        outputs = [out['generated_text'] for out in self.pipe(text, **kwargs)]
        return outputs

def find_control_options(tokenizer):
    special_tokens = [token_id.split('_')[1:] for token_id in tokenizer.additional_special_tokens]
    control_options = defaultdict(list)
    for tok_details in special_tokens:
        div, div_type, value = tok_details
        control_options[div.lower() + '_' + div_type.lower()].append(int(value))
    return dict(control_options)

def find_missing_control_options(tokenizer):
    control_options = find_control_options(tokenizer)
    missing_options = defaultdict(list)
    for div, observed_options in control_options.items():
        for expected_val in range(0, 101, 5):
            if expected_val not in observed_options:
                missing_options[div].append(expected_val)
    return dict(missing_options)

def nearest_value(v, values):
    arr = np.asarray(values)
    i = (np.abs(arr - v)).argmin()
    return arr[i]