# Assignment: Research Engineer in Natural Language Processing
## RISE Research Institutes of Sweden

This repo contains code that finetunes a language model on the English data of MultiNERD (<https://aclanthology.org/2022.findings-naacl.60.pdf>)

Two finetuned systems can be produced: 
* system **A** is finetuned on all given NER tags
* system **B** is finetuned on PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), and ANIMAL(ANIM) only

The models are evaluated with precision, recall and F1 from seqeval.

### To replicate:

An available GPU is assumed.

First, install the required packages:
```
pip install -r requirements.txt
```

To finetune a model, run train.py followed by 
A or B to choose the dataset labels to finetune the model on, and 
an optional output directory. 
For example, to finetune a system A model, i.e. on all the NER tags:
```
python train.py A ./a_results
```
The default output directory is `./{a,b}_results`.
Hyperparameters and other configurations can be set in `conf.yml`.

To test, run test.py followed by 
A or B to choose the dataset labels for the test set, and 
an optional input directory of your model. For example, to test a model on all the NER tags:
```
python test.py A ./a_results
```
The default input directory of the model is `./{a,b}_results`.

A Jupyter notebook version is also available.