import random
from random import sample

import numpy as np
import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from .submod.submodopt import SubmodularOpt
from .submod.submodular_funcs import trigger_dips

class DiPSParaphraser:

    def __init__(self, augmenter="dips", num_outputs=3, seed=42, verbose=False):
        self.augmenter = augmenter
        self.num_outputs = num_outputs
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        assert augmenter in ["dips", "random", "diverse_beam", "beam"]
        if self.verbose:
            choices = ["dips", "random", "diverse_beam", "beam"]
            print(
                "The base paraphraser being used is Backtranslation - Generating {} candidates based on {}\n".format(
                    num_outputs, augmenter
                )
            )
            print(
                "Primary options for augmenter : {}. \n".format(str(choices))
            )
            print(
                "Default: augmenter='dips', num_outputs=3. Change using DiPSParaphraser(augmenter=<option>, num_outputs=<num_outputs>)\n"
            )
            print("Starting to load English to German Translation Model.")

        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(
            name_en_de
        ).to(self.device)

        if self.verbose:
            print("Completed loading English to German Translation Model.")
            print("Starting to load German to English Translation Model.")

        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(
            name_de_en
        ).to(self.device)

        if self.verbose:
            print("Completed loading German to English Translation Model.")

        self.augmenter = augmenter
        if self.augmenter == "dips":
            if self.verbose:
                print("Loading word2vec gensim model. Please wait...")
            trigger_dips()
            if self.verbose:
                print("Completed loading word2vec gensim model.")
        self.num_outputs = num_outputs

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt").to(self.device)
        outputs = self.model_en_de.generate(input_ids).cpu()
        decoded = self.tokenizer_en_de.decode(
            outputs[0], skip_special_tokens=True
        )
        if self.verbose:
            print(decoded)
        return decoded

    def generate_diverse(self, en: str):
        # try:
        de = self.en2de(en)
        if self.augmenter == "diverse_beam":
            en_new = self.generate_diverse_beam(de)
        else:
            en_new = self.select_candidates(de, en)
        # except Exception as e:
        #     if self.verbose:
        #         print("Returning Default due to Run Time Exception", e)
        #     en_new = [en for _ in range(self.num_outputs)]
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
        if self.augmenter == "dips":
            try:
                subopt = SubmodularOpt(decoded, sentence)
                subopt.initialize_function(0.4, a1=0.5, a2=0.5, b1=1.0, b2=1.0)
                predicted_outputs = list(
                    subopt.maximize_func(self.num_outputs)
                )
            except Exception as e:
                if self.verbose:
                    print("Error in SubmodularOpt: {}".format(e))
                predicted_outputs = decoded[: self.num_outputs]
        elif self.augmenter == "random":
            predicted_outputs = sample(decoded, self.num_outputs)
        else:  # Fallback to top n points in beam search
            predicted_outputs = decoded[: self.num_outputs]

        if self.verbose:
            print(predicted_outputs)

        return predicted_outputs

    def generate_diverse_beam(self, sentence: str):
        input_ids = self.tokenizer_de_en.encode(sentence, return_tensors="pt").to(self.device)

        try:
            outputs = self.model_de_en.generate(
                input_ids,
                num_return_sequences=self.num_outputs,
                num_beam_groups=2,
                num_beams=self.num_outputs,
            ).cpu()
        except Exception:
            outputs = self.model_de_en.generate(
                input_ids,
                num_return_sequences=self.num_outputs,
                num_beam_groups=1,
                num_beams=self.num_outputs,
            ).cpu()

        predicted_outputs = [
            self.tokenizer_de_en.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        if self.verbose:
            print(predicted_outputs)

        return predicted_outputs

    def generate(self, sentence: str):
        candidates = self.generate_diverse(sentence)
        return candidates

    def __call__(self, text):
        out = self.generate(text)
        # clean out memory
        del self.model_en_de, self.model_de_en
        return out

if __name__ == '__main__':

    text = 'She sells seashells by the seashore.'

    transform_fn = DiPSParaphraser(augmenter='dips', num_outputs=3)
    paraphrases = transform_fn.generate(text)
    print(paraphrases)
