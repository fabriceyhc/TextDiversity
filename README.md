# TextDiversity
 
## Installation

```
pip install textdiversity
```

Some of the text diversity measures rely on software that must be installed separately. For your particular OS, follow the instructions below:

### Windows

1. Phoneme Diversity
   - Phonemizer
      - Visit: https://github.com/espeak-ng/espeak-ng/releases
      - Install either `espeak-ng-X64.msi` or `espeak-ng-X86.msi` (This project used Release 1.51)
      - Add Phonemizer environement variable ([[reference]](https://github.com/bootphon/phonemizer/issues/44))
         - Navigate: Windows Key > Edit System Environment Variables > Environment Variables
         - Add New Variable:
            - Variable Name: `PHONEMIZER_ESPEAK_LIBRARY`
            - Variable Value: `C:\Program Files\eSpeak NG\libespeak-ng.dll`
2. Syntactical Diversity (non-core functionality)
   - Dependency Parse Tree Visualization
      - `UNKOWN SETUP STEPS`

### Linux

1. Phoneme Diversity
   - Phonemizer
      - `sudo apt-get install espeak-ng`
2. Syntactical Diversity
   - Dependency Parse Tree Visualization (non-core functionality)
      - `sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config`
    

## Example Usage

```python
import pandas as pd
from textdiversity import (
    TokenSemantics, DocumentSemantics, AMR, # semantics
    DependencyParse, ConstituencyParse,     # syntactical
    PartOfSpeechSequence,                   # morphological
    Rhythmic                                # phonological
)

corpus1 = ['one massive earth', 'an enormous globe', 'the colossal world'] # unique words
corpus2 = ['basic human right', 'you were right', 'make a right']          # lower unigram diversity

metrics = [
    TokenSemantics(), DocumentSemantics(), AMR(), 
    DependencyParse(), ConstituencyParse(), 
    PartOfSpeechSequence(), 
    Rhythmic()
]

results = []
for metric in metrics:
    results.append({
        "metric": metric.__class__.__name__,
        "corpus1": metric(corpus1),
        "coprus2": metric(corpus2)
    })

df = pd.DataFrame(results)
```

|    | metric               |   corpus1 |   coprus2 |
|---:|:---------------------|----------:|----------:|
|  0 | TokenSemantics       |   7.42473 |   7.99136 |
|  1 | DocumentSemantics    |   1.18850 |   1.65927 |
|  2 | AMR                  |   2.76379 |   1.71087 |
|  3 | DependencyParse      |   1.00000 |   1.88204 |
|  4 | ConstituencyParse    |   1.00001 |   1.88989 |
|  5 | PartOfSpeechSequence |   1.17621 |   1.80000 |
|  6 | Rhythmic             |   1.36364 |   1.81230 |
