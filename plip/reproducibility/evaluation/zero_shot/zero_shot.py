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

    def zero_shot_classification(self, image_embeddings, text_embeddings, unique_labels, target_labels, model_name, train_ds_name=None, test_ds_name=None, preprocess = None, mode='Single', model=None, text_error=None, adversarial=False):
        
        score = image_embeddings.dot(text_embeddings.T)
        predictions = [unique_labels[np.argmax(i)] for i in score]

        ## Calculating the total number of parameters in a model
        if model_name == "plip":
            total_params = sum(p.numel() for p in model.model.parameters())
        else:
            total_params = sum(p.numel() for p in model.parameters())

        
        ## creating designated folder for the results
        if mode == 'ensemble':
            folder_path = f"/home/roba/Histopathology_Benchmark/plip/ensemble/{test_ds_name}/"
            
        else:

            folder_path = f"/home/roba/Histopathology_Benchmark/plip/results/{test_ds_name}/"

        if adversarial:
            folder_path = f"/home/roba/Histopathology_Benchmark/plip/advers_results/{test_ds_name}/"
        os.makedirs(folder_path, exist_ok=True)


        test_metrics = eval_metrics(target_labels, predictions, model_name, train_ds_name, test_ds_name, score, unique_labels, dest=folder_path, error=text_error)
        train_metrics = eval_metrics(target_labels, predictions, model_name, train_ds_name, test_ds_name, score, unique_labels, dest=folder_path, error=text_error)
        ece = calibration.main(score, target_labels, test_metrics["Accuracy"], test_metrics["instances"], model_name, test_ds_name, unique_labels, folder_path, error_name=text_error)
        train_metrics["ece"] = ece
        test_metrics["ece"] = ece
        test_metrics["params"] = total_params
        cm = confusion_matrix(target_labels, predictions)
        # Plot confusion matrix without displaying it
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(target_labels), yticklabels=np.unique(target_labels))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
    
        # Save the plot as an image without displaying it
        if adversarial:
            plt.savefig(os.path.join(folder_path,f"confusion_matrix_{model_name}_adverse_8.png"), bbox_inches='tight', pad_inches=0)
        else:
            # print("In the correct path for errors")    
            plt.savefig(os.path.join(folder_path,f"confusion_matrix_{model_name}_{text_error}_3.png"), bbox_inches='tight', pad_inches=0)
            # plt.savefig(os.path.join(folder_path,f"confusion_matrix_{model_name}_{text_error}_3.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        dit = {"train_metrics":train_metrics, "test_metrics":test_metrics}
        if os.path.isdir(folder_path) == False:
            os.mkdir(folder_path)
        if adversarial:
            f= open(os.path.join(folder_path,f"{model_name}_adverse_0.3_zero.json"), 'w')
        else:
            f= open(os.path.join(folder_path,f"{model_name}_{text_error}_3_zero.json"), 'w')
        json.dump(dit, f)
        f.close()
        test_metrics["split"] = "test"
        train_metrics["split"] = "train"

        import pickle
        if adversarial:
            with open(f"{folder_path}/pickle_adverse_0.1.pkl", "wb") as f:
                pickle.dump({"target" : target_labels,
                            "predictions" : predictions}, f)
            exit()
        else:
            with open(f"{folder_path}/pickle_{text_error}_3.pkl", "wb") as f:
                pickle.dump({"target" : target_labels,
                            "predictions" : predictions}, f)
            exit()
        logging.info(f"ZeroShot Done")

        return train_metrics, test_metrics
