# Training and evaluating a Named Entity Recognition model for English with MultiNERD

This repo contains code that finetunes a language model on the English data of [MultiNERD](<https://aclanthology.org/2022.findings-naacl.60.pdf>)

Two finetuned systems can be produced: 
* system **A** is finetuned on all given NER tags
* system **B** is finetuned on PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), and ANIMAL(ANIM) only

The models are evaluated with precision, recall and F1 from [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py).

### To replicate:

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
an output directory for `metrics.json` and `results.json`. 
For example, to test a model on all the NER tags:
```
python test.py A ./model_dir ./eval_output_dir
```

A Jupyter notebook version is also available.

Update (26/1/2024): Optional CLI arguments added:
```
-d , --device              Device to train/test the model on (default='cuda')
--disable_tqdm             Disable tqdm
-p, --output_predictions   (Only for test.py) Output predictions on test set as results.json
```

*An assignment for RISE.*