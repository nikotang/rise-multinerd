from collections import defaultdict

# set up look-up dictionaries for the model
# label2id dictionary from https://huggingface.co/datasets/Babelscape/multinerd
label2id = {
  "O": 0,
  "B-PER": 1,
  "I-PER": 2,
  "B-ORG": 3,
  "I-ORG": 4,
  "B-LOC": 5,
  "I-LOC": 6,
  "B-ANIM": 7,
  "I-ANIM": 8,
  "B-BIO": 9,
  "I-BIO": 10,
  "B-CEL": 11,
  "I-CEL": 12,
  "B-DIS": 13,
  "I-DIS": 14,
  "B-EVE": 15,
  "I-EVE": 16,
  "B-FOOD": 17,
  "I-FOOD": 18,
  "B-INST": 19,
  "I-INST": 20,
  "B-MEDIA": 21,
  "I-MEDIA": 22,
  "B-MYTH": 23,
  "I-MYTH": 24,
  "B-PLANT": 25,
  "I-PLANT": 26,
  "B-TIME": 27,
  "I-TIME": 28,
  "B-VEHI": 29,
  "I-VEHI": 30,
}
id2label = {v:k for k,v in label2id.items()}
label_list = list(label2id.keys())

# system B: only 5 categories
label_list_B = ['O',
    'B-PER',
    'I-PER',
    'B-ORG',
    'I-ORG',
    'B-LOC',
    'I-LOC',
    'B-ANIM',
    'I-ANIM',
    'B-DIS',
    'I-DIS'
    ]

# the same label-id correspondence cannot be kept because the training process only allows label ids of range(0:number of classifications)
label2id_B = {l:i for i,l in enumerate(label_list_B)}
id2label_B = {v:k for k,v in label2id_B.items()}

# map the tokenized dataset to the simpler set of labels
# map system A ids to system B ids, set the rest to 0
A2B_mapping = defaultdict(lambda:0, {label2id[label]:label2id_B[label] for label in label_list_B})
A2B_mapping[-100] = -100        # for special tokens and trailing subtokens of NER entities
