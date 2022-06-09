# TextDiversity
 
## Installation

```
pip install -r requirements.txt
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