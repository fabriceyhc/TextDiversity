import benepar
import nltk
import spacy

from .sowreap.parse_utils import Sentence
from .sowreap.reap_utils import reapModel
from .sowreap.sow_utils import sowModel

class SowReapParaphraser:

    def __init__(self, seed=0, num_outputs=1):
        self.sow = sowModel("tanyagoyal/paraphrase-sow", num_outputs)
        self.reap = reapModel("tanyagoyal/paraphrase-reap", num_outputs)
        self.nlp = spacy.load("en_core_web_sm")
        try:
            nltk.data.find("models/benepar_en3")
        except LookupError:
            benepar.download("benepar_en3")

        if spacy.__version__.startswith("2"):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        self.num_outputs = num_outputs

    def generate(self, sentence: str):
        # use benepar model to retrieve the constituency parse (as a string)
        parse = list(self.nlp(sentence).sents)[0]._.parse_string
        sentence_parsed = Sentence(parse)
        reorderings = self.sow.get_reorderings(sentence_parsed)
        if len(reorderings) == 0:
            return []
        transformations = self.reap.get_transformations(
            sentence_parsed, reorderings
        )
        return transformations

    def __call__(self, text):
        return self.generate(text)


if __name__ == "__main__":

    tf = SowReapParaphraser(num_outputs=3)

    sentence = (
        "the company withdrew its application on the 196th day its submission."
    )
    output = tf.generate(sentence)
    print(output)
