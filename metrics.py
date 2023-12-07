import evaluate
import numpy as np

seqeval = evaluate.load('seqeval')

def set_compute_metrics(label_list):
  def compute_metrics(p):
    nonlocal label_list     # available with python>=3.x
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
      [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
      [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
      'precision': results['overall_precision'],
      'recall': results['overall_recall'],
      'f1': results['overall_f1'],
      'accuracy': results['overall_accuracy'],
    }
  return compute_metrics