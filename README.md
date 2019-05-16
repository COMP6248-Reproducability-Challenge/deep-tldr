# deep-tldr
This repository contains our implementation of th ICLR paper [A Deep Reinforced Model for Abstractive Summarization](https://openreview.net/forum?id=HkAClQgA-). 

## Report
The report can be found [here](ReproductionReport.pdf), and should be read before running. 

## Usage
The vast majority of the relevant code is in `main.py`. All major parameters are contained in `config.py` (such as the learning rate, batch size, and which dataset to use). The D1 dummy dataset is hardcoded into `dataset.py`, while D2 is contained within the `\data_fake\` folder. The real CNN/DM dataset is too large to store, but can be found [here](https://github.com/rohithreddy024/Text-Summarizer-Pytorch), and should be placed into a new `\data\` folder. 

Use `python [main.py]` to run the code. 

