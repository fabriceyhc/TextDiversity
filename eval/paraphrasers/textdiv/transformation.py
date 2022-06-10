import random
import numpy as np
import torch
from random import sample
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from .submod.submodopt import SubmodularOpt


class TextDiversityParaphraser:

    def __init__(self, num_outputs=3, seed=42, verbose=False):
        self.num_outputs = num_outputs
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.verbose:
            print("Starting to load English to German Translation Model...")

        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(name_en_de).to(self.device)

        if self.verbose:
            print("Completed loading English to German Translation Model.")
            print("Starting to load German to English Translation Model...")

        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(name_de_en).to(self.device)

        if self.verbose:
            print("Completed loading German to English Translation Model.")

        if self.verbose:
            print("Initializing textdiv instances. Please wait...")
        self.subopt = SubmodularOpt()
        if self.verbose:
            print("Completed initializing textdiv instances.")

        self.num_outputs = num_outputs

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt").to(self.device)
        outputs = self.model_en_de.generate(input_ids).cpu()
        decoded = self.tokenizer_en_de.decode(outputs[0], skip_special_tokens=True)
        if self.verbose:
            print(decoded)
        return decoded

    def generate_diverse(self, en: str):
        de = self.en2de(en)
        en_new = self.select_candidates(de, en)
        return en_new

    def select_candidates(self, input: str, sentence: str):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt").to(self.device)
        outputs = self.model_de_en.generate(
            input_ids,
            num_return_sequences=self.num_outputs * 10,
            num_beams=self.num_outputs * 10,
        ).cpu()

        predicted_outputs = []
        decoded = [
            self.tokenizer_de_en.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        self.subopt.V = decoded
        self.subopt.v = sentence

        self.subopt.initialize_function(
                    lam = 0.5, 
                    w_toksim = 1.0,
                    w_docsim = 1.0,
                    w_posdiv = 1.0, 
                    w_rhydiv = 1.0, 
                    w_phodiv = 1.0, 
                    w_depdiv = 1.0)

        predicted_outputs = list(self.subopt.maximize_func(self.num_outputs))

        return predicted_outputs

    def generate(self, sentence: str):
        candidates = self.generate_diverse(sentence)
        return candidates

    def __call__(self, text):
        print('__call__')
        out = self.generate(text)
        # clean out memory
        del self.model_en_de, self.model_de_en
        return out


if __name__ == '__main__':

    text = 'She sells seashells by the seashore.'

    transform_fn = TextDiversityParaphraser(num_outputs=3, verbose=True)
    paraphrases = transform_fn.generate(text)
    print(paraphrases)
