from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, disable_progress_bar

import gc
import torch
import argparse
from pathlib import Path

from conf import load_conf
from tokenization import process_dataset, system_B_labels
import labels
from metrics import set_compute_metrics

parser = argparse.ArgumentParser(description='''Train a language model on MultiNERD English examples. 
                                 System A trains on all NER labels;
                                 system B trains only on PER, ORG, LOC, DIS and ANIM. ''')
parser.add_argument('system', choices=['A', 'B'], help='System "A" or "B"')
parser.add_argument('conf', type=Path, help='Path to the config YAML file')
parser.add_argument('output_dir', type=Path, help='Path to the output directory')
parser.add_argument('--disable_tqdm', action='store_true', default=False, help='Disable tqdm')
parser.add_argument('-d', '--device', type=str, help='Device to train the model on', default='cuda')
args = parser.parse_args()

def main(args):
    conf = load_conf(args.conf)

    # fetch tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(conf['model_name_or_path'])
    if args.disable_tqdm:
        disable_progress_bar()
    dataset = load_dataset('Babelscape/multinerd')

    # filter dataset to only contain English data
    eng_dataset = dataset.filter(lambda batch: [lang=='en' for lang in batch['lang']], batched=True)

    # tokenize the dataset and adjust the labels accordingly
    tokenized_eng = eng_dataset.map(process_dataset(tokenizer), batched=True)

    # set data collator, pads to len(longest example of the batch)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # get labels and lookup tables
    label2id = labels.label2id
    id2label = labels.id2label
    label_list = labels.label_list

    if args.system == 'B':
        label2id = labels.label2id_B
        id2label = labels.id2label_B
        label_list = labels.label_list_B
        tokenized_eng = tokenized_eng.map(system_B_labels, batched=True)

    #finetune
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # set arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,         
        num_train_epochs=conf['num_epochs'],
        max_steps=conf['max_steps'],                        # overrides training epochs
        per_device_train_batch_size=conf['per_gpu_train_batch_size'],
        per_device_eval_batch_size=conf['per_gpu_eval_batch_size'],  
        warmup_steps=conf['warmup_steps'],
        learning_rate=conf['learning_rate'],
        weight_decay=conf['weight_decay'],
        log_level='info',
        logging_dir=f'./{args.system}_logs', 
        logging_steps=conf['log_steps'],
        evaluation_strategy='steps', 
        eval_steps=conf['eval_steps'],
        save_steps=conf['save_steps'],
        save_total_limit=conf['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',      # determine 'best' according to eval loss
        greater_is_better=False,
        dataloader_drop_last=True,              # stops when what remains is less than a batch when training by steps
        disable_tqdm=args.disable_tqdm
    )

    # load the model
    model = AutoModelForTokenClassification.from_pretrained(conf['model_name_or_path'], 
                                                            num_labels=len(id2label), 
                                                            id2label=id2label, 
                                                            label2id=label2id,
                                                            hidden_dropout_prob=conf['hidden_dropout_prob'],
                                                            ).to(args.device)

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=tokenized_eng['train'], 
        eval_dataset=tokenized_eng['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=set_compute_metrics(label_list),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=conf['early_stopping_patience'])]      # checks n more steps before early stopping
    )

    trainer.train()

    trainer.save_model()

if __name__ == "__main__":
    main(args=args)
