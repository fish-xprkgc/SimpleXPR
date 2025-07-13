Here is XPR-KGC, explainable path reward model for knowledge graph completion.
The main part is path contrastive learning and have 2 extra modules ,which are llm-relation module
and skip module, you can change the parameters in .sh file to modify these modules on/off.


The first step of this project is preprocess the dataset into igraph and csv format for later process. 
In this stage, we use deepseek to hel[ us generate neural language relation in dataset, so if you want 
to preprocess data , you need to modify the api-key in line 103 of preprocess.py, or you can start at 
second stage by download the total file in data dir.

For original WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).

1、data preprocess

to preprocess WN18RR dataset, you can run this script in root dir: bash scripts/preprocess.sh WN18RR

to preprocess FB15k237 dataset, you can run this script in root dir: bash scripts/preprocess.sh FB15k237

2、train the model

