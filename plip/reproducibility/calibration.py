# Code adapted from https://github.com/Jonathan-Pearce/calibration_library

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.special import softmax
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm




class CELoss(object):

    def compute_bin_boundaries(self, probabilities=np.array([])):

        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            bin_n = int(self.n_data / self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)

            for i in range(0, self.n_bins):
                bin_boundaries = np.append(bin_boundaries,
                                           probabilities_sort[i * bin_n])
            bin_boundaries = np.append(bin_boundaries, 1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def get_probabilities(self, output, labels,unique_labels, logits):
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = np.array(labels)
        self.confidences = np.max(self.probabilities, axis=1)
        indcs = list(torch.max(torch.tensor(output), 1)[1])
        self.predictions = np.array(unique_labels[[indcs[i]for i in range(len(indcs))]])
        self.accuracies = np.equal(self.predictions, labels)
        

    def compute_bins(self, index=None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:, index]
            accuracies = self.acc_matrix[:, index]

        for i, (bin_lower,
                bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = np.greater(confidences, bin_lower.item()) * np.less_equal(
                confidences, bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])


class MaxProbCELoss(CELoss):

    def loss(self, output, labels,unique_labels, n_bins=50, logits=True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels,unique_labels, logits)
        super().compute_bins()


#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels,unique_labels, n_bins=50, logits=True):
        super().loss(output, labels,unique_labels, n_bins, logits)
        return np.dot(self.bin_prop, self.bin_score)


def save_fig(model_name, dataset, bin_acc, ece, n_bins=50, bar_color=None):
    mpl.rcParams['font.size'] = 14
    yellow = '#FFFF00'

    if bar_color is None:
        bar_color = '#fc8e62'

    delta = 1.0 / n_bins
    x = np.arange(0, 1, delta)
    mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
    error = np.abs(np.subtract(mid, bin_acc))

    plt.figure(figsize=(6, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.bar(x,
            bin_acc,
            color=bar_color,
            width=delta,
            align='edge',
            edgecolor='k',
            label='Predictions')
    plt.bar(x,
            error,
            bottom=np.minimum(bin_acc, mid),
            color=yellow,
            alpha=0.5,
            width=delta,
            align='edge',
            edgecolor='r',
            hatch='/',
            label='Gap')

    ident = [0.0, 1.0]
    plt.plot(ident, ident, linestyle='--', color='tab:grey')

    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')

    plt.xticks(np.arange(0.2, 1.1, 0.2), fontsize=17)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=17)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.text(0.5,
             0.95,
             f'ECE: {ece:.4f}',
             ha='center',
             va='center',
             transform=plt.gca().transAxes,
             fontsize=14)

    plt.title(f'{model_name}  {dataset}')

    plt.tight_layout()

    results_path = "/home/roba.majzoub/research/fall2024/Histopathology_Benchmark/plip/reproducibility/calibration"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    plt.savefig(os.path.join(results_path,f"{model_name}_{dataset}_calibration.pdf"))


    return plt


def main(logits, labels, passed_accuracy, total, model_name, test_ds_name, unique_labels):

    np.random.seed(0)

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    output_probs = F.softmax(logits, dim=1)
    probs, indcs = torch.max(output_probs, 1)
    indcs = list(indcs)
    unique_labels = torch.tensor(unique_labels)
    predicted = unique_labels[[indcs[i]for i in range(len(indcs))]]
    correct = (predicted == labels).sum().item()

    

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    print(f'Calculated Accuracy {100 * correct / total}')
    print(f'Passed Accuracy {passed_accuracy}')
    print(total)
    ece_criterion = ECELoss()
    logits_np = logits.cpu().numpy()
    labels_np = labels.cpu().numpy()
    ece = ece_criterion.loss(logits_np, labels_np, unique_labels, 50)
    print(f'{model_name} ECE: {round(ece, 4)}')

    save_fig(model_name, test_ds_name, ece_criterion.bin_acc, ece)
    return ece


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inference script for Beyond ImageNet Accuracy project')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='imagenet',
                        help='Type of dataset to use (imagenet or imagenet_r)')
    parser.add_argument('--models',
                        nargs='+',
                        default=[
                            "deit3_21k", "convnext_base_21k", "vit_clip",
                            "convnext_clip"
                        ],
                        help='List of models to analyze')
    parser.add_argument('--pretrained_dir',
                        default="../pretrained",
                        type=str,
                        help='Directory containing pretrained models')
    args = parser.add_argument('--dataset_root', type=str)
    args = parser.parse_args()
    main(args)