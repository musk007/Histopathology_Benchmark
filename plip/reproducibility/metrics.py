import numpy as np
import os
import json
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, matthews_corrcoef, accuracy_score, classification_report, balanced_accuracy_score
import torchmetrics
import torch
import sys

def retrieval_metrics(y_target, y_predictions):
    p_10 = 0
    p0 = 0

    for t, predictions in zip(y_target, y_predictions):
        if t in predictions[:10]:
            p_10 += 1
        if t in predictions[:50]:
            p0 += 1

    return {"p@10": p_10/len(y_target), "p@50": p0/len(y_target)}

# ### adding the ECE (Estimated Confirdence Calibration)
# def expected_calibration_error(samples, true_labels, M):
#     # uniform binning with M bins




def eval_metrics(y_true, y_pred, model_name, train_ds_name, test_ds_name, finetune_test, y_pred_proba = None, average_method='weighted'):
    assert len(y_true) == len(y_pred)
    if y_pred_proba is None:
        auroc = np.nan
    elif len(np.unique(y_true)) > 2:
        print('Multiclass AUC is not currently available.')
        auroc = np.nan
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auroc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred, average = average_method)
    print(classification_report(y_true, y_pred))
    ########################################### ADDED
    if finetune_test == True:
        folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/finetune_evaluation/trained_on_{train_ds_name}/{test_ds_name}"
        save_f = os.path.join(folder_path,f"{model_name}_per_class_performance_{test_ds_name}.json")
    else:
        if test_ds_name == None:
            raise ValueError("test_ds_name is required")
        # folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/caption_results/{test_ds_name}/original_cap"
        # folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/caption_results/{test_ds_name}/new_cap/"
        folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/results/{test_ds_name}/"
        save_f = os.path.join(folder_path,f"{model_name}_per_class_performance_{test_ds_name}_all.json")
    if os.path.isdir(folder_path) == False:
        os.mkdir(folder_path)
    f = open(save_f, 'w')
    json.dump(classification_report(y_true, y_pred), f)
    f.close()
    ###########################################
    precision = precision_score(y_true, y_pred, average = average_method)
    recall = recall_score(y_true, y_pred, average = average_method)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==1:
           tp += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           fp += 1
        if y_true[i]==y_pred[i]==0:
           tn += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           fn += 1
    if (tp+fn) == 0: sensitivity = np.nan
    else: sensitivity = tp/(tp+fn) # recall
    if (tn+fp) == 0: specificity = np.nan
    else: specificity = tn/(tn+fp)
    if (tp+fp) == 0: ppv = np.nan
    else: ppv = tp/(tp+fp) # precision or positive predictive value (PPV)
    if (tn+fn) == 0: npv = np.nan
    else: npv = tn/(tn+fn) # negative predictive value (NPV)
    if (tp+tn+fp+fn) == 0: hitrate = np.nan
    else: hitrate = (tp+tn)/(tp+tn+fp+fn) # accuracy (ACC)
    performance = {'Accuracy': acc,
                   'Balanced Accuracy': b_acc,
                   'AUC': auroc,
                   'WF1': f1,
                   'precision': precision,
                   'recall': recall,
                   'mcc': mcc,
                   'tp': tp,
                   'fp': fp,
                   'tn': tn,
                   'fn': fn,
                   'sensitivity': sensitivity,
                   'specificity': specificity,
                   'ppv': ppv,
                   'npv': npv,
                   'hitrate': hitrate,
                   'instances' : len(y_true)}
    return performance