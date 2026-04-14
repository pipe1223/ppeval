import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_precision_recall_f1(ground_truth, predictions):
    #sigle_ground_truth_answer
    gt_flat = [gt for gt in ground_truth] 

    ##multi_ground_truth_answer
    #gt_flat = [gt[0] for gt in ground_truth]
    accuracy = accuracy_score(gt_flat, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(gt_flat, predictions, average='weighted')
    return accuracy, precision, recall, f1

def evaluate_vqa(ground_truth, predictions):
    return calculate_precision_recall_f1(ground_truth, predictions)