import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.metrics import specificity_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, precision_recall_fscore_support

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
import numpy as np
from collections import defaultdict

def sample_acc(y_true_bin, y_pred_bin):
    y_true_bin = np.asarray(y_true_bin)
    y_pred_bin = np.asarray(y_pred_bin)
    return (y_true_bin == y_pred_bin).all(axis=1).mean()

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def single_label(y_true_bin, y_pred_bin, class_names):
    # Calculate accuracy, precision, recall, and F1 score
    accuracy_avg = accuracy_score(y_true_bin, y_pred_bin)
    precision_macro = precision_score(y_true_bin, y_pred_bin, average='macro')
    recall_macro = recall_score(y_true_bin, y_pred_bin, average='macro')
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro')  

    precision_mi = precision_score(y_true_bin, y_pred_bin, average=None)
    recall_mi = recall_score(y_true_bin, y_pred_bin, average=None)
    f1_mi = f1_score(y_true_bin, y_pred_bin, average=None)

    micro_metrics = {}
    for class_name, p, r, f in zip(class_names, precision_mi, recall_mi, f1_mi):
        micro_metrics[class_name] = {'Precision':p*100, 'Recall':r*100, 'F1-score':f*100}
        
    return accuracy_avg*100, precision_macro*100, recall_macro*100, f1_macro*100, micro_metrics

# def multi_label(y_true_bin, y_pred_bin, class_names):
#     print (class_names)
#     print ("y_true_bin:", y_true_bin)
#     print ("y_pred_bin:", y_pred_bin)
#     accuracy = sample_acc(y_true_bin, y_pred_bin)#Accuracy(y_true_bin, y_pred_bin)
#     precision = precision_score(y_true_bin, y_pred_bin, average='samples')
#     recall = recall_score(y_true_bin, y_pred_bin, average='samples')
#     f1 = f1_score(y_true_bin, y_pred_bin, average='samples')

#     precision_mi = precision_score(y_true_bin, y_pred_bin, average=None)
#     recall_mi = recall_score(y_true_bin, y_pred_bin, average=None)
#     f1_mi = f1_score(y_true_bin, y_pred_bin, average=None)

#     micro_metrics = {}
#     for class_name, p, r, f in zip(class_names, precision_mi, recall_mi, f1_mi):
#         if not ',' in class_name:
#             micro_metrics[class_name] = {'Precision':p*100, 'Recall':r*100, 'F1-score':f*100}

#     return accuracy*100, precision*100, recall*100, f1*100, micro_metrics

def multi_label(y_true_bin, y_pred_bin, label_names):
    y_true_bin = np.asarray(y_true_bin)
    y_pred_bin = np.asarray(y_pred_bin)

    # sample-wise (your original)
    accuracy = sample_acc(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)

    # per-label (NOT micro)
    p_lbl, r_lbl, f_lbl, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )

    per_label_metrics = {
        name: {'Precision': float(p*100), 'Recall': float(r*100), 'F1-score': float(f*100)}
        for name, p, r, f in zip(label_names, p_lbl, r_lbl, f_lbl)
    }

    return accuracy*100, precision*100, recall*100, f1*100, per_label_metrics


def sample_acc(y_true, y_pred):
    sample_accuracies = []
    for true, pred in zip(y_true, y_pred):
        # Convert binary vectors to sets of indices where value is 1
        true_set = set(idx for idx, val in enumerate(true) if val == 1)
        pred_set = set(idx for idx, val in enumerate(pred) if val == 1)
        
        # Calculate intersection and union
        if len(true_set) > 0:  # Avoid division by zero
            intersection = len(true_set & pred_set)
            union = len(true_set | pred_set)
            sample_accuracies.append(intersection / union)
        else:
            sample_accuracies.append(0)  # No ground truth labels
    
    # Calculate average accuracy
    average_accuracy = sum(sample_accuracies) / len(sample_accuracies)
    print(f"Sample-Level Accuracy: {average_accuracy:.2f}")
    return average_accuracy


def evaluate_classification(y_true, y_pred, class_names=''):
    # Check if the task is multi-label based on presence of commas
    is_multi_label = any(',' in yt for yt in y_true)# or any(',' in yp for yp in y_pred)

    if is_multi_label:

        # 转换为多标签格式
        y_true_sets = [set(labels.split(', ')) for labels in y_true]
        y_pred_sets = [set(labels.split(', ')) for labels in y_pred]

        gt_per_class = defaultdict(int)
        
        # Iterate through each set in y_true_sets
        for label_set in y_true_sets:
            for class_name in label_set:
                gt_per_class[class_name] += 1
                
        gt_per_class = dict(gt_per_class)
        
        # # 统计所有可能的标签
        # all_labels = list(set.union(*y_true_sets, *y_pred_sets))
    
        # # 将标签集转为多标签二进制矩阵
        # y_true_bin = [[1 if label in true else 0 for label in all_labels] for true in y_true_sets]
        # y_pred_bin = [[1 if label in pred else 0 for label in all_labels] for pred in y_pred_sets]



        # print("Multi-label task detected.")
        # accuracy_avg, precision_macro, recall_macro, f1_macro, micro_metrics = multi_label(y_true_bin, y_pred_bin,class_names)


        all_labels = list(set.union(*y_true_sets, *y_pred_sets))
        y_true_bin = [[1 if label in true else 0 for label in all_labels] for true in y_true_sets]
        y_pred_bin = [[1 if label in pred else 0 for label in all_labels] for pred in y_pred_sets]

        accuracy_avg, precision_macro, recall_macro, f1_macro, micro_metrics = multi_label(y_true_bin, y_pred_bin, all_labels)
        
        return  accuracy_avg, precision_macro, recall_macro, f1_macro, micro_metrics, gt_per_class

    else:
        
        # Convert each string of classes into a list of classes, removing any extra spaces
        y_true = [set(yt.strip().split(', ')) for yt in y_true]
        y_pred = [set(yp.strip().split(', ')) for yp in y_pred]
        
        # Find all unique classes in y_pred that are not in class_names
        unique_pred_classes = set().union(*y_pred)
        extra_classes = unique_pred_classes - set(class_names)
        
        # Rename any extra classes to 'dummy'
        if extra_classes:
            print(f"Found classes in y_pred not in class_names: {extra_classes}. Renaming to 'dummy'.")
            y_pred = [{('dummy' if cls in extra_classes else cls) for cls in yp} for yp in y_pred]
            class_names = class_names + ['dummy']  # Add 'dummy' class to class_names
    
        # Convert to one-hot encoded format using MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=class_names)
        
        # Transform y_true and y_pred to a binary indicator matrix
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        print("Single-label task detected.")
        #ROC_single_label(y_true_bin, y_pred_bin,class_names) #for drow roc
        
        gt_counter_per_class = {}
        
        for class_name in class_names:
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            gt_counter_per_class[class_name] = 1

        accuracy_avg, precision_macro, recall_macro, f1_macro, micro_metrics = single_label(y_true_bin, y_pred_bin,class_names)
        
        return accuracy_avg, precision_macro, recall_macro, f1_macro, micro_metrics, gt_counter_per_class




def ROC_single_label(y_true_bin, y_pred_bin, class_names, output_file='roc_curve.png'):
    """
    Generate and save ROC curve for single-label classification.
    
    Parameters:
    y_true_bin : array-like, shape = [n_samples]
        True binary labels (integer valued, not one-hot encoded)
    y_pred_bin : array-like, shape = [n_samples, n_classes]
        Predicted probabilities for each class
    class_names : list, shape = [n_classes]
        Names of the classes
    output_file : str, optional
        Path to save the ROC curve image (default: 'roc_curve.png')
    """
    # Binarize the true labels (convert to one-hot encoding)
    y_true_bin = label_binarize(y_true_bin, classes=np.arange(len(class_names)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_names)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save the figure instead of showing it
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory