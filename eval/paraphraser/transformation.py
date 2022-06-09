import random
import numpy as np
import torch
from random import sample
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from .submod.submodopt import SubmodularOpt


class TextDiversityParaphraser:

    def __init__(self, augmenter="textdiv", num_outputs=3, seed=42, verbose=False):
        self.augmenter = augmenter
        self.num_outputs = num_outputs
        self.verbose = verbose

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        assert augmenter in ["textdiv", "random", "diverse_beam", "beam"]
        if self.verbose:
            choices = ["textdiv", "random", "diverse_beam", "beam"]
            print(
                "The base paraphraser being used is Backtranslation - Generating {} candidates based on {}\n".format(
                    num_outputs, augmenter
                )
            )
            print("Primary options for augmenter : {}. \n".format(str(choices)))
            print(
                "Default: augmenter='textdiv', num_outputs=3. Change using DiverseParaphrase(augmenter=<option>, num_outputs=<num_outputs>)\n"
            )
            print("Starting to load English to German Translation Model...")

        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(name_en_de)

        if self.verbose:
            print("Completed loading English to German Translation Model.")
            print("Starting to load German to English Translation Model...")

        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(name_de_en)

        if self.verbose:
            print("Completed loading German to English Translation Model.")

        self.augmenter = augmenter
        if self.augmenter == "textdiv":
            if self.verbose:
                print("Initializing textdiv instances. Please wait...")
            self.subopt = SubmodularOpt()
            if self.verbose:
                print("Completed initializing textdiv instances.")
        self.num_outputs = num_outputs

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt")
        outputs = self.model_en_de.generate(input_ids)
        decoded = self.tokenizer_en_de.decode(outputs[0], skip_special_tokens=True)
        if self.verbose:
            print(decoded)
        return decoded

    def generate_diverse(self, en: str):
        try:
            de = self.en2de(en)
            if self.augmenter == "diverse_beam":
                en_new = self.generate_diverse_beam(de)
            else:
                en_new = self.select_candidates(de, en)
        except Exception:
            if self.verbose:
                print("Returning Default due to Run Time Exception")
            en_new = [en for _ in range(self.num_outputs)]
        return en_new

    def select_candidates(self, input: str, sentence: str):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
        outputs = self.model_de_en.generate(
            input_ids,
            num_return_sequences=self.num_outputs * 5,
            num_beams=self.num_outputs * 5,
        )

        predicted_outputs = []
        decoded = [
            self.tokenizer_de_en.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        if self.augmenter == "textdiv":
            try:
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
        input_ids = self.tokenizer_de_en.encode(sentence, return_tensors="pt")

        try:
            outputs = self.model_de_en.generate(
                input_ids,
                num_return_sequences=self.num_outputs,
                num_beam_groups=2,
                num_beams=self.num_outputs,
            )
        except:
            outputs = self.model_de_en.generate(
                input_ids,
                num_return_sequences=self.num_outputs,
                num_beam_groups=1,
                num_beams=self.num_outputs,
            )

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


if __name__ == '__main__':

    text = 'She sells seashells by the seashore.'

    transform_fn = TextDiversityParaphraser(augmenter='textdiv', num_outputs=3)
    paraphrases = transform_fn.generate(text)
    print(paraphrases)
