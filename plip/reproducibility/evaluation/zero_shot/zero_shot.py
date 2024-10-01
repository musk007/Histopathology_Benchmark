import numpy as np
from reproducibility.metrics import eval_metrics
from reproducibility import calibration
import logging
import warnings
import json
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys

class ZeroShotClassifier:

    def __init__(self):
        warnings.filterwarnings('ignore')
        pass

    def zero_shot_classification(self, image_embeddings, text_embeddings, unique_labels, target_labels, model_name, train_ds_name=None, test_ds_name=None, finetune_test=False):
        

        score = image_embeddings.dot(text_embeddings.T)
        predictions = [unique_labels[np.argmax(i)] for i in score]
        

        ## creating designated folder for the results
        if finetune_test == True:
            folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/finetune_evaluation/trained_on_{train_ds_name}/{test_ds_name}"
            
        else:
            # folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/caption_results/{test_ds_name}/original_cap"
            # folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/caption_results/{test_ds_name}/new_cap/"
            folder_path = f"/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/results/{test_ds_name}/"
        os.makedirs(folder_path, exist_ok=True)


        test_metrics = eval_metrics(target_labels, predictions, model_name, train_ds_name, test_ds_name, finetune_test)
        train_metrics = eval_metrics(target_labels, predictions, model_name, train_ds_name, test_ds_name, finetune_test)
        ece = calibration.main(score, target_labels, test_metrics["Accuracy"], test_metrics["instances"], model_name, test_ds_name, unique_labels)
        train_metrics["ece"] = ece
        test_metrics["ece"] = ece
        cm = confusion_matrix(target_labels, predictions)
        # Plot confusion matrix without displaying it
        # plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(target_labels), yticklabels=np.unique(target_labels))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
    
        # Save the plot as an image without displaying it
        
        plt.savefig(os.path.join(folder_path,f"confusion_matrix_{model_name}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        dit = {"train_metrics":train_metrics, "test_metrics":test_metrics}
        if os.path.isdir(folder_path) == False:
            os.mkdir(folder_path)
        f= open(os.path.join(folder_path,f"{model_name}_zero.json"), 'w')
        json.dump(dit, f)
        f.close()
        test_metrics["split"] = "test"
        train_metrics["split"] = "train"

        import pickle
        with open(f"{folder_path}/pickle.pkl", "wb") as f:
            pickle.dump({"target" : target_labels,
                         "predictions" : predictions}, f)
        exit()
        logging.info(f"ZeroShot Done")

        return train_metrics, test_metrics
