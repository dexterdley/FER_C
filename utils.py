#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:28:29 2020

@author: dexter
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F
import pdb
import torch.nn as nn
import random
from ECE import *
import torch
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

MSE = nn.MSELoss()
criterion = F.binary_cross_entropy_with_logits

def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_weights(m, bias_const=0.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, bias_const)

def load_pretrained(model, model_path):
    pretrained = torch.load(model_path)
    pretrained_state_dict = pretrained['state_dict']
    model_state_dict = model.state_dict()
    loaded_keys = 0
    total_keys = 0
    for key in pretrained_state_dict:
        if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
            pass
        else:    
            model_state_dict[key] = pretrained_state_dict[key]
            total_keys+=1
            if key in model_state_dict:
                loaded_keys+=1
    print("Loaded params num:", loaded_keys)
    print("Total params num:", total_keys)
    model.load_state_dict(model_state_dict, strict = False)

    return model

def add_noise(df, noise_ratio, num_classes):
    random.seed(0)
    
    corrupted_len = int(noise_ratio/100 * len(df) )
    idx = random.choices(df.index, k=corrupted_len)

    corrupted_labels = random.choices( range(num_classes), k=corrupted_len)
    df['expression'].loc[idx] = corrupted_labels
            
    return df


def get_weights(df, num_classes):
    '''
    Compute the distribution of each class
    '''
    
    classes = np.arange(num_classes)

    class_weights = {}

    for k in classes:
        class_idx = np.where(df['expression'] == k)[0]
        
        class_weights[k] = len(class_idx)
       
    
    class_weights = list(class_weights.values())
    
    max_class = np.max(class_weights) # following the biggest class
    
    return class_weights, max_class/class_weights

def smooth_one_hot(onehot, num_classes: int, epsilon=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    
    out = (1 - epsilon) * onehot + epsilon/num_classes
    
    return out

def compute_CCC(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : tensor
        Ground truth vector
    y_pred : tensor
        Precicted values

    Returns
    -------
    CCC, a float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.

    Reference code: https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
    '''
    cor = np.corrcoef(y_true,y_pred)[0,1] #return correlation covariance matrix, take [0][1] as pearson coefficient
    
    mean_true, var_true, sd_true = y_true.mean(), y_true.var(), y_true.std()
    mean_pred, var_pred, sd_pred = y_pred.mean(), y_pred.var(), y_pred.std()
 
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2
    
    return numerator/denominator, cor 

def get_closer_class(predictions, centroids):
    
    distances = []
    
    for k in range(len(centroids)):
        
        dist = torch.sqrt((predictions[0] - centroids[k][0]).pow(2) + (predictions[1] - centroids[k][1]).pow(2))
        distances.append(dist.item())

    return distances

def get_va_labels(centroids, targets):

    '''
    Inputs: centroids and ground truth VA points
    Outputs: Tensor of mixed soft labels
    '''

    distances = np.array([get_closer_class(point, centroids) for point in targets]) #compute euclidean dist
    distances = np.max(distances, axis=1).reshape(len(distances),1)/ distances
    va_onehot = torch.tensor(distances/ np.sum(distances, axis=1).reshape(len(distances),1) )

    #sim = torch.tensor([get_similar_class(point, centroids) for point in targets]) #compute cosine similarity

    return va_onehot

def convert_to_onehot(array, num_classes=8):
    
    B = len(array) #batch size
    
    out = np.zeros((B, num_classes))
    out[range(B), array.to(cpu)] = 1
    
    return torch.FloatTensor(out)

def mse(A, B, class_weights):
    
    if class_weights is None:
        
        y = torch.square(A - B)
        
    else:
        
        y = torch.square(A - B) * class_weights
    
    out = torch.mean(y)
    return out

def cosine_loss(pred, y, class_weights):

    epsilon = torch.tensor([1e-8]).to(device)
    
    numerator = torch.sum(pred * y, dim=1)
    denominator = torch.max( torch.norm(pred, dim=1) * torch.norm(y, dim=1), epsilon) # computes cos(theta)
    
    if class_weights is None:
        
        loss = (1 - numerator/denominator)
    else:
    
        loss = (1 - numerator/denominator) * class_weights
    
    return torch.mean(loss) 
    

def mae(pred, y, class_weights):
    
    B, k = pred.shape #batch by number of classes
    
    zeros = torch.zeros(pred.shape).to(device)
    
    if class_weights is None:
        
        out = torch.max(zeros, (pred - y))
        
    else:
        
        out = torch.max(zeros, (pred - y)) * class_weights.reshape(B, 1)
    
    out = torch.sum(out, dim=1) 
    
    return torch.mean(out)

def negative_log_likelihood(pred, y, class_weights):

    ce_loss = y * torch.log(pred.clamp(1e-6)) 
    
    if class_weights is None:
        
        ce_loss = -torch.sum(ce_loss, dim=1)
        
    else:
        
        ce_loss = -torch.sum(ce_loss, dim=1)  * class_weights #changed 
        
    return torch.mean(ce_loss)


def get_top_classes(vec, rank, top_k):
    new_vec = np.zeros(rank.shape)

    for i in range(top_k):
    
        top_idx = np.where(i == rank)
        new_vec[top_idx] = vec[top_idx]
        
    return new_vec

def update_CALS(loader, model, CALS, epoch):
    
    model.eval()
    cpu = torch.device('cpu')
    CALS.reset_update_lambd(epoch)
    
    with torch.no_grad():
        
        for count, (x, y, labels, GMM_label) in enumerate(loader):
            
            x = x.to(device = device)
            y = y.to(device = device)
            labels = labels.to(device = device)
            GMM_label = GMM_label.to(device = device)

            #forward
            _, score, logits = model(x)
            CALS.update_lambd(logits, epoch)
            penalty, constraint = CALS.get(logits)

        CALS.set_lambd(epoch)
        CALS.update_rho(epoch)

    print("Updated Lam: %.3f" %CALS.get_lambd_metric()[1], "Rho: %.3f" %CALS.get_rho_metric()[1])


def evaluate_model(loader, model, num_classes, T=1, algo=None):
    """
    Parameters
    ----------
    loader : Dataloader
        Validation dataset
    model : TYPE
        DESCRIPTION.
    
    Using only 3-channel RGB model
    
    Returns
    -------
    RMSE and CCC values for both Arousal and Valence predictions,
    Classification report
    """
    
    model.eval()
    cpu = torch.device('cpu')
    
    with torch.no_grad():
        
        ce_losses = []

        pred_probs = []
        pred_all = []
        GT_all = []
        
        num_correct_classes = 0
        num_samples_classes = 0 #4000
        num_samples = 0 #4500

        labels_list = []
        predictions_list = []
        confidence_vals_list = []
        
        brier_error_list = []

        for count, (x, labels, GMM_label) in enumerate(loader):
            
            x = x.to(device = device)
            labels = labels.to(device = device)
            GMM_label = GMM_label.to(device = device)

            #forward
            if algo == "DMUE":
                rand = torch.randint(low=0, high=num_classes, size=labels.shape).to('cuda')
                logits, _, _, _ = model(x, rand, 'normal') #not supposed to have knowledge of labels

            elif algo == 'RUL':
                logits = model(x, labels, phase='test')
                
            else:
                _, _, logits = model(x)

            class_pred = F.softmax(logits/T, dim=1)
            classes = torch.argmax(class_pred, dim=1)
            
            pred_all.append(classes)
            GT_all.append(labels)
            
            onehot = convert_to_onehot(labels, num_classes).to(device) #convert classes to onehot representation
            CE_loss = negative_log_likelihood(class_pred, onehot, None)
            ce_losses.append(CE_loss.item())
            
            num_correct_classes += torch.sum(torch.eq(classes, labels )).item()
            num_samples += len(x)
            num_samples_classes += len(labels)

            brier_error = torch.square(onehot - class_pred).mean() # measure the brier loss of misclassified samples
            brier_error_list.append(brier_error.item() )

            confidence_vals, predictions = torch.max(class_pred, dim=1)

            labels_list.extend(labels.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().detach().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().detach().numpy().tolist())

            pred_probs.append(class_pred)

        pred_probs = torch.cat(pred_probs)
        pred_all = torch.cat(pred_all)
        GT_all = torch.cat(GT_all)
        
        #pdb.set_trace()
        scores = metrics.classification_report(GT_all.to(cpu), pred_all.to(cpu), digits = 4)
        F1 = metrics.f1_score(GT_all.to(cpu), pred_all.to(cpu), average='macro')
        accuracy = num_correct_classes/num_samples_classes

        avg_ce_loss = sum(ce_losses)/len(ce_losses)

        #compute ECE here
        ECE = expected_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=15)
        MCE = maximum_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=15)
        brier = sum(brier_error_list)/len(brier_error_list)

        adaece = adaptive_expected_calibration_error(confidence_vals_list, predictions_list, labels_list ).item()
        cece = classwise_calibration_error(pred_probs.cpu(), labels_list, num_classes ).item()
        KS_error_max = KSE_func(confidence_vals_list, predictions_list, labels_list)


        return avg_ce_loss, scores, accuracy, F1, ECE, cece, adaece, KS_error_max

def evaluate_GMM(loader, model, fuse_factor):
    """
    Parameters
    ----------
    loader : Dataloader
        Validation dataset
    model : TYPE
        DESCRIPTION.
    
    Using only 3-channel RGB model
    
    Returns
    -------
    RMSE and CCC values for both Arousal and Valence predictions,
    Classification report
    """
    
    model.eval()
    cpu = torch.device('cpu')
    
    with torch.no_grad():
        
        error_V, error_A = 0.0, 0.0
        CCC_V = []
        CCC_A = []
        corr_V = []
        corr_A = []
        
        mse_losses = []
        ce_losses = []
        cos_losses = []

        pred_all = []
        GT_all = []
        
        num_correct_classes = 0
        num_samples_classes = 0 #4000
        num_samples = 0 #4500

        labels_list = []
        predictions_list = []
        confidence_vals_list = []
        
        NLL_error_list = []
        brier_error_list = []
        
        for count, (x, y, labels, GMM_label) in enumerate(loader):
            
            x = x.to(device = device)
            y = y.to(device = device)
            labels = labels.to(device = device)
            GMM_label = GMM_label.to(device = device)

            #forward
            _, score, logits = model(x)

            class_pred = F.softmax(logits, dim=1)
            classes = torch.argmax(class_pred, dim=1)
            
            pred_all.append(classes)
            GT_all.append(labels)
            
            valence_pred = score[:,0]
            arousal_pred = score[:,1]
            
            MSE_loss = MSE(score, y)
                        
            onehot = convert_to_onehot(labels).to(device) #convert classes to onehot representation
            
            beta = fuse_factor
            va_onehot = beta * onehot + (1 - beta) * GMM_label
            y_idx = np.where(onehot.cpu() ==1) 
            
            CE_loss = negative_log_likelihood(class_pred, va_onehot, None)
            cos_loss = cosine_loss(class_pred, va_onehot, None)

            mse_losses.append(MSE_loss.item())
            ce_losses.append(CE_loss.item())
            cos_losses.append(cos_loss.item())
            
            error_V += torch.sum( torch.square(valence_pred - y[:, 0]) )
            error_A += torch.sum( torch.square(arousal_pred - y[:, 1]) )
            
            ccc_v, corr_v = compute_CCC( y[:, 0].to(cpu), valence_pred.to(cpu))
            ccc_a, corr_a = compute_CCC( y[:, 1].to(cpu), arousal_pred.to(cpu))
             
            CCC_V.append(ccc_v)
            CCC_A.append(ccc_a)
            corr_V.append(corr_v)
            corr_A.append(corr_a)
            
            #pdb.set_trace()
            num_correct_classes += torch.sum(torch.eq(classes, labels )).item()
            num_samples += len(x)
            num_samples_classes += len(labels)

            compare_array = torch.eq(class_pred.argmax(1), labels )

            wrong_idx = torch.where(compare_array == 0) # index of misclassified samples
            wrong_probs = class_pred[wrong_idx] #predicted probabilites of misclassified samples
            
            NLL_error = -torch.sum(onehot[wrong_idx] * torch.log(wrong_probs), dim=1).mean() # measure the NLL of misclassified samples
            brier_error = torch.square(onehot[wrong_idx] - wrong_probs).mean() # measure the brier loss of misclassified samples
            
            NLL_error_list.append(NLL_error.item() )
            brier_error_list.append(brier_error.item() )

            confidence_vals, predictions = torch.max(class_pred, dim=1)

            labels_list.extend(labels.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().detach().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().detach().numpy().tolist())

                
        error_V = error_V.to(cpu)
        error_A = error_A.to(cpu)
        
        mse_V = error_V/ num_samples
        mse_A = error_A/ num_samples
        
        CCC_Valence = np.mean(CCC_V)
        CCC_Arousal = np.mean(CCC_A)
        
        Corr_V = np.mean(corr_V)
        Corr_A = np.mean(corr_A)
        
        pred_all = torch.cat(pred_all)
        GT_all = torch.cat(GT_all)
        
        #pdb.set_trace()
        scores = metrics.classification_report(GT_all.to(cpu), pred_all.to(cpu), digits = 4)
        #conf_mat = metrics.confusion_matrix(y_true=GT_all.to(cpu), y_pred=pred_all.to(cpu))
        F1 = metrics.f1_score(GT_all.to(cpu), pred_all.to(cpu), average='macro')
        accuracy = num_correct_classes/num_samples_classes

        avg_mse_loss = sum(mse_losses)/len(mse_losses)
        avg_ce_loss = sum(ce_losses)/len(ce_losses)
        avg_cos_loss = sum(cos_losses)/len(cos_losses)

        #compute ECE here
        ECE = expected_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=10)
        MCE = maximum_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=10)
        NLL = sum(NLL_error_list)/len(NLL_error_list)
        brier = sum(brier_error_list)/len(brier_error_list)

        return avg_mse_loss, avg_ce_loss, avg_cos_loss, np.sqrt(mse_V), np.sqrt(mse_A), CCC_Valence, CCC_Arousal, Corr_V, Corr_A, scores, accuracy, F1, ECE, MCE, NLL, brier

def save_checkpoint(state, epoch, expt_name):
    print("==> Checkpoint saved")
    
    if not os.path.exists('./models/' + expt_name):
        os.makedirs('./models/' + expt_name)
        
    outfile = './models/' + expt_name + '/' + str(epoch) + '_' + expt_name + '.pth.tar'
    torch.save(state, outfile)
    
def load_checkpoint(model, weight_file):
    print("==> Loading Checkpoint: " + weight_file)
    
    if torch.cuda.is_available() == False:
        checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(weight_file)
        
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_pretrained(model, model_path):
    pretrained = torch.load(model_path)
    pretrained_state_dict = pretrained['state_dict']
    model_state_dict = model.state_dict()
    loaded_keys = 0
    total_keys = 0
    for key in pretrained_state_dict:
        if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
            pass
        else:    
            model_state_dict[key] = pretrained_state_dict[key]
            total_keys+=1
            if key in model_state_dict:
                loaded_keys+=1
    print("Loaded params num:", loaded_keys)
    print("Total params num:", total_keys)
    model.load_state_dict(model_state_dict, strict = False)

    return model