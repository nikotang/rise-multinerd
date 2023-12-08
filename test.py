from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

from sys import argv

from metrics import set_compute_metrics
from tokenizer import process_dataset, system_B_labels
import labels

if len(argv) != 4:
    raise Exception('Give 3 arguments: system, model directory and output directory. ')

system, model_dir, output_dir = argv[1:4]
if system not in ['A', 'B']:
    raise ValueError('Fine-tuning system is either "A" or "B". ') 

def main(system, model_dir, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    label_list = labels.label_list

    # load and process the dataset
    #if train.py was run before, Huggingface should have cached the dataset
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
        per_device_eval_batch_size = 64
    )

    tester = Trainer(
                model = test_model,
                args = test_args,
                tokenizer=tokenizer,
                data_collator = data_collator,
                compute_metrics = set_compute_metrics(label_list)
    )

    tester.evaluate(eval_dataset=tokenized_eng)

if __name__ == "__main__":
    main(system=system, model_dir=model_dir, output_dir=output_dir)
