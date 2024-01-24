from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, disable_progress_bar

import argparse
from pathlib import Path
import json
from scipy.special import softmax
from tqdm import tqdm

from metrics import set_compute_metrics
from tokenization import process_dataset, system_B_labels
import labels

parser = argparse.ArgumentParser(description='''Test a language model finetuned on MultiNERD English examples. 
                                 System A trains on all NER labels;
                                 system B trains only on PER, ORG, LOC, DIS and ANIM. ''')
parser.add_argument('system', choices=['A', 'B'], help='System "A" or "B"')
parser.add_argument('model_dir', type=Path, help='Path to the model directory')
parser.add_argument('output_dir', type=Path, help='Path to the output directory')
parser.add_argument('-p', '--output_predictions', action='store_true', help='Output predictions on test set as results.json')
parser.add_argument('--disable_tqdm', action='store_true', default=False, help='Disable tqdm')
parser.add_argument('-d', '--device', type=str, help='Device to test the model on', default='cuda')
args = parser.parse_args()

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    label_list = labels.label_list

    # load and process the dataset
    #if train.py was run before, Huggingface should have cached the dataset
    if args.disable_tqdm:
        disable_progress_bar()
    dataset = load_dataset('Babelscape/multinerd', split='test')

    # filter dataset to only contain English data
    eng_dataset = dataset.filter(lambda batch: [lang=='en' for lang in batch['lang']], batched=True)

    # tokenize the dataset and adjust the labels accordingly
    tokenized_eng = eng_dataset.map(process_dataset(tokenizer), batched=True)

    if args.system == 'B':
        label_list = labels.label_list_B
        tokenized_eng = tokenized_eng.map(system_B_labels, batched=True)

    test_model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(args.device)

    test_args = TrainingArguments(
        output_dir = args.output_dir,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = 64,
        log_level='info',
        disable_tqdm=args.disable_tqdm
    )

    tester = Trainer(
                model = test_model,
                args = test_args,
                tokenizer=tokenizer,
                data_collator = data_collator,
                compute_metrics = set_compute_metrics(label_list, per_type=True)
    )

    test_results = tester.predict(test_dataset=tokenized_eng)

    if args.output_predictions:
        # List words and ner tags predicted
        probabilities = softmax(test_results.predictions, axis=-1)
        predictions = test_results.predictions.argmax(axis=-1)
        results = []
        for i, prediction in enumerate(tqdm(predictions, disable=args.disable_tqdm)):
            result = []
            idx = 0
            while idx < len(tokenized_eng['input_ids'][i]):
                pred = prediction[idx]
                word_idx = tokenized_eng['word_ids'][i][idx]
                if pred != 0 and word_idx != None:
                    label = test_model.config.id2label[pred]
                    result.append(
                        {
                        'word_idx': word_idx,
                        'entity': label,
                        'score': float(probabilities[i][idx][pred]), 
                        'word': eng_dataset['tokens'][i][word_idx],
                        }
                    )
                    while tokenized_eng['word_ids'][i][idx] == word_idx:
                        idx += 1
                else:
                    idx += 1
            results.append(result)
            
        with open(f'{args.output_dir}/results.json', 'w') as fout:
            json.dump(results, fout, indent=4)

    with open(f'{args.output_dir}/metrics.json', 'w') as fout:
        json.dump(test_results.metrics, fout, indent=4)

if __name__ == "__main__":
    main(args=args)
