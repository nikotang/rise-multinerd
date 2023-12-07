import numpy as np

from labels import A2B_mapping

def tokenize_and_align_labels(examples, tokenizer):
  '''
  Tokenize Dataset or DatasetDict, and set labels for non-first subtokens as -100 to ignore loss calculation.
  '''
  tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True) # the examples are already split into words
  labels = []
  for i, label in enumerate(examples['ner_tags']):
    word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100
      if word_idx is None:
        label_ids.append(-100)
      elif word_idx != previous_word_idx:  # Only label the first token of a given word
        label_ids.append(label[word_idx])
      else:
        label_ids.append(-100)
      previous_word_idx = word_idx
    labels.append(label_ids)
  tokenized_inputs['labels'] = labels
  return tokenized_inputs

def apply_mapping(label):
  return A2B_mapping[label]

def system_B_labels(example):
  vmap = np.vectorize(apply_mapping)
  for i, tags in enumerate(example['ner_tags']):
    example['ner_tags'][i] = vmap(tags)
  for i, tags in enumerate(example['labels']):
    example['labels'][i] = vmap(tags)
  return example