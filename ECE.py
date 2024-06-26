'''
Metrics to measure calibration of a trained deep neural network.
References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.

Accessory code taken from: https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py

@article{mukhoti2020calibrating,
  title={Calibrating Deep Neural Networks using Focal Loss},
  author={Mukhoti, Jishnu and Kulharia, Viveka and Sanyal, Amartya and Golodetz, Stuart and Torr, Philip HS and Dokania, Puneet K},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}

'''
import math
import matplotlib.pyplot as plt
import pdb
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
plt.rcParams.update({'font.size': 40})

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return ece


def soft_populate_bins(confs, preds, GT, num_bins=10):
    labels_confs, labels = GT.max(1)
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        label_conf = labels_confs[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] += 1
        bin_dict[binn][CONF] += confidence
        bin_dict[binn][ACC] += (label_conf if (label == prediction) else 1 - label_conf)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict

def soft_expected_calibration_error(confs, preds, GT, num_bins=15):
    bin_dict = soft_populate_bins(confs, preds, GT, num_bins)
    num_samples = len(confs)
    sece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        sece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return sece

def maximum_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        ce.append(abs(bin_accuracy - bin_confidence))
    return max(ce)


def reliability_plot(confs, preds, labels, title, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    '''
    plt.figure(figsize=(10, 8))  # width:20, height:3
    #plt.title(title)
    plt.bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    #plt.legend()
    plt.show()
    '''
    return y

def bin_strength_plot(confs, preds, labels, title, num_bins=15):
    '''
    Method to draw a plot for the number of samples in each confidence bin.
    '''
    #fontsz = 30
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples))
        y.append(n)
    '''
    plt.figure(figsize=(10, 8))  # width:20, height:3
    #plt.title(title)
    plt.bar(bns, y, align='edge', width=0.05,
            color='lightcyan', edgecolor='black', linewidth=2, alpha=1, label='Percentage samples',)
    plt.ylabel('Percentage of samples')
    plt.xlabel('Predicted probability')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    '''
    return y

def histedges_equalN(confs, num_bins=15):
    npt = len(confs)
    return np.interp(np.linspace(0, npt, num_bins + 1),
            np.arange(npt),
            np.sort(confs))

def adaptive_expected_calibration_error(confs, preds, labels, num_bins=15):
    
    confs = torch.tensor(confs)
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)

    accuracies = preds.eq(labels)
    n, bin_boundaries = np.histogram(confs, histedges_equalN(confs, num_bins))
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
        
    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        # Calculated |confidence - accuracy| in each bin
        in_bin = confs.gt(bin_lower.item()) * confs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confs_in_bin = confs[in_bin].mean()
            ece += np.abs(avg_confs_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def soft_adaptive_expected_calibration_error(confs, preds, labels, num_bins=15):
    
    confs = torch.tensor(confs)
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)

    accuracies = torch.zeros(len(labels))

    for i in range(len(labels)):
        if preds[i] == labels[i].argmax():

            accuracies[i] = labels[i].max()
        else:
            accuracies[i] = 1 - labels[i].max()
    

    n, bin_boundaries = np.histogram(confs, histedges_equalN(confs, num_bins))
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        # Calculated |confidence - accuracy| in each bin
        in_bin = confs.gt(bin_lower.item()) * confs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confs_in_bin = confs[in_bin].mean()
            ece += np.abs(avg_confs_in_bin - accuracy_in_bin) * prop_in_bin

    return ece  



def classwise_calibration_error(probs, labels, num_classes, num_bins=15):
    labels = torch.tensor(labels)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    per_class_sce = None

    for i in range(num_classes):
        class_confidences = probs[:, i]
        class_sce = torch.zeros(1)
        labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            
            in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = labels_in_class[in_bin].float().mean()
                avg_confidence_in_bin = class_confidences[in_bin].mean()
                class_sce += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        if (i == 0):
            per_class_sce = class_sce
        else:
            per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

    sce = torch.mean(per_class_sce)
    return sce


def KSE_func(confidence_vals_list, predictions_list, labels_list):
    '''
    Reference code: https://github.com/kartikgupta-at-anu/spline-calibration/blob/83c85a4302a85f0a1d7f64ab779c5af28fb7e96f/cal_metrics/KS.py#L4
    Paper: CALIBRATION OF NEURAL NETWORKS USING SPLINES ICLR2019
    '''

    scores = np.array(confidence_vals_list)
    preds = np.array(predictions_list)
    labels = np.array(predictions_list) == np.array(labels_list)

    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    nsamples = len(confidence_vals_list)
    integrated_accuracy = np.cumsum(labels) / nsamples
    integrated_scores   = np.cumsum(scores) / nsamples
    KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))

    return KS_error_max


def soft_KSE_func(confidence_vals_list, predictions_list, labels_list):
    '''
    Reference code: https://github.com/kartikgupta-at-anu/spline-calibration/blob/83c85a4302a85f0a1d7f64ab779c5af28fb7e96f/cal_metrics/KS.py#L4
    Paper: CALIBRATION OF NEURAL NETWORKS USING SPLINES ICLR2019
    '''

    scores = np.array(confidence_vals_list)
    preds = np.array(predictions_list)
    #accuracies = np.array(predictions_list) == np.array(labels_list)

    accuracies = np.zeros(len(labels_list))
    for i in range(len(labels_list)):
        if predictions_list[i] == labels_list[i].argmax():

            accuracies[i] = labels_list[i].max()
        else:
            accuracies[i] = 1 - labels_list[i].max()

    order = scores.argsort()
    scores = scores[order]
    accuracies = accuracies[order]

    nsamples = len(confidence_vals_list)
    integrated_accuracy = np.cumsum(accuracies) / nsamples
    integrated_scores   = np.cumsum(scores) / nsamples
    KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))

    return KS_error_max