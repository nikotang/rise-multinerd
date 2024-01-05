# Training and evaluating a Named Entity Recognition model for English with MultiNERD

This repo contains code that finetunes a language model on the English data of [MultiNERD](<https://aclanthology.org/2022.findings-naacl.60.pdf>)

Two finetuned systems can be produced: 
* system **A** is finetuned on all given NER tags
* system **B** is finetuned on PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), and ANIMAL(ANIM) only

The models are evaluated with precision, recall and F1 from [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py).

### To replicate:

An available GPU is assumed.

First, install the required packages:
```
pip install -r requirements.txt
```
To finetune a model, run `train.py` followed by 
`A` or `B` to choose the dataset labels to finetune the model on, 
directory to the `conf.yml` file, and
an output directory. 
For example, to finetune a system A model, i.e. on all the NER tags:
```
python train.py A ./conf.yml ./model_output_dir
```
Hyperparameters and other configurations can be set in `conf.yml`.

To test, run `test.py` followed by 
`A` or `B` to choose the dataset labels for the test set, 
a directory to your model, and 
an output directory for `results.json` and `metrics.json`, that contains evaluation results and metrics respectively. 
For example, to test a model on all the NER tags:
```
python test.py A ./model_dir ./eval_output_dir
```

A Jupyter notebook version is also available.

Update (5/1/2024): Test output now includes the individual words and NER tags predicted.

*An assignment for RISE.*