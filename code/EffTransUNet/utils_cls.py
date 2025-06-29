import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os


def calculate_metrics(preds, labels, num_classes):
    results = {}


    results['accuracy'] = accuracy_score(labels, preds)
    results['macro_f1'] = f1_score(labels, preds, average='macro')


    class_report = {}
    for class_id in range(num_classes):
        precision = np.sum((preds == class_id) & (labels == class_id)) / np.sum(preds == class_id)
        recall = np.sum((preds == class_id) & (labels == class_id)) / np.sum(labels == class_id)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        class_report[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return results, class_report

