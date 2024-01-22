from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, disable_progress_bar

import argparse
from pathlib import Path
import json

from metrics import set_compute_metrics
from tokenization import process_dataset, system_B_labels
import labels

parser = argparse.ArgumentParser(description='''Test a language model finetuned on MultiNERD English examples. 
                                 System A trains on all NER labels;
                                 system B trains only on PER, ORG, LOC, DIS and ANIM. ''')
parser.add_argument('system', choices=['A', 'B'], help='System "A" or "B"')
parser.add_argument('model_dir', type=Path, help='Path to the model directory')
parser.add_argument('output_dir', type=Path, help='Path to the output directory')
args = parser.parse_args()

def main(system, model_dir, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    label_list = labels.label_list

    # load and process the dataset
    #if train.py was run before, Huggingface should have cached the dataset
    disable_progress_bar()
    dataset = load_dataset('Babelscape/multinerd', split='test')

    # filter dataset to only contain English data
    eng_dataset = dataset.filter(lambda batch: [lang=='en' for lang in batch['lang']], batched=True)

    # tokenize the dataset and adjust the labels accordingly
    tokenized_eng = eng_dataset.map(process_dataset(tokenizer), batched=True)

    if system == 'B':
        label_list = labels.label_list_B
        tokenized_eng = tokenized_eng.map(system_B_labels, batched=True)

    test_model = AutoModelForTokenClassification.from_pretrained(model_dir).to('cuda')

    test_args = TrainingArguments(
        output_dir = output_dir,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = 64,
        log_level='info',
        disable_tqdm=True
    )

    tester = Trainer(
                model = test_model,
                args = test_args,
                tokenizer=tokenizer,
                data_collator = data_collator,
                compute_metrics = set_compute_metrics(label_list, per_type=True)
    )

    results = tester.evaluate(eval_dataset=tokenized_eng)

    with open(f'{output_dir}/results.json', 'w') as fout:
        json.dump(results, fout, indent=4)

if __name__ == "__main__":
    main(system=args.system, model_dir=args.model_dir, output_dir=args.output_dir)
