import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from sklearn.metrics import auc
import torch.nn.functional as F
import pdb

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

from maxent_newton_solver import *


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.eps = 1e-7

    def forward(self, p, y, weights):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))
        loss = -torch.sum(loss @ weights, dim=1)

        #pdb.set_trace()
        return torch.mean(loss)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-8):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps

    def forward(self, p, y, weights):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))
        
        if weights != None:
            loss *= weights

        if self.gamma > 0:
            # Focal loss
            focal = (1 - p).pow(self.gamma)
            loss *= focal

        #pdb.set_trace()
        loss = -torch.sum(loss, dim=1)
        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

class InvFocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-8):
        super(InvFocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        
    def forward(self, p, y, weights):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))

        if weights != None:
            loss *= weights
            
        inv_focal = -(1 + p).pow(self.gamma)
        loss *= inv_focal

        loss = torch.sum(loss, dim=1)
        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

class PolyLoss(nn.Module):
    def __init__(self, epsilon=-1, eps=1e-8):
        super(PolyLoss, self).__init__()

        self.eps = eps
        self.epsilon = epsilon
        
    def forward(self, p, y, weights):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """

        batch_sz = y.shape[0]
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))
        
        if weights != None:
            loss *= weights

        poly = loss + self.epsilon * (1 - y*p)
        loss = -torch.sum(poly, dim=1)

        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

class MaxEntLoss(nn.Module):
    def __init__(self, ratio, constraints, gamma=2, num_classes=10, eps=1e-8):
        super(MaxEntLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        self.num_classes = num_classes
        self.ratio = ratio.to(device)

        self.constraints = constraints
        
        self.x = torch.tensor(range(num_classes), dtype=float).to(device)
        self.target_mu = torch.sum(self.ratio * self.x).to(device)
        #print(self.lam_1)
    def forward(self, p, y, weights):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """

        batch_sz = y.shape[0]
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))

        mu = torch.sum(p * self.x, dim=1)
        local_mu = torch.sum(y*self.x, dim=1)

        if weights != None:
            loss *= weights

        # Focal loss
        if self.gamma > 0:
            focal = (1 - p).pow(self.gamma)
            loss *= focal

        if self.constraints == 6: #Exponential Distribution (Mean constraint)
            self.lam_1 = solve_mean_lagrange(self.x, self.target_mu, p)
            mu_loss = torch.abs(mu - self.target_mu)

            loss = -torch.sum(loss, dim=1) + self.lam_1 * mu_loss

        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

def my_auc(x, y):
    direction =1 
    dx = torch.diff(x)

    if torch.any(dx < 0):
        if torch.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing : {}.".format(x))

    return direction * torch.trapz(y, x)

class AUAvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is probabilities from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]

    The reference code is from: https://github.com/IntelLabs/AVUC/blob/main/src/avuc_loss.py
    """
    def __init__(self, beta=1):
        super(AUAvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10
        self.focal = FocalLoss(gamma=0)

    def entropy(self, prob):
        return -torch.sum(prob * torch.log(prob.clamp(self.eps)), dim=1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def forward(self, probs, y, weights, type=0):

        confidences, predictions = torch.max(probs, 1)
        labels = torch.argmax(y)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        #th_list = np.linspace(0, 1, 21)
        th_list = torch.linspace(0, 1, 21, requires_grad=True).to(device)
        umin, umax = torch.min(unc), torch.max(unc)

        avu_list = []
        unc_list = []

        for t in th_list:
            unc_th = umin + (torch.tensor(t, device=labels.device) * (umax - umin))
            
            n_ac = torch.zeros(1, device=labels.device)
            n_ic = torch.zeros(1, device=labels.device)
            n_au = torch.zeros(1, device=labels.device)
            n_iu = torch.zeros(1, device=labels.device)
            
            #Use masks and logic operators to compute the 4 differentiable proxies
            n_ac_mask = torch.logical_and(labels == predictions, unc <= unc_th)
            n_ac = torch.sum( confidences[n_ac_mask] * (1 - torch.tanh(unc[n_ac_mask]) ) )

            n_au_mask = torch.logical_and(labels == predictions, unc > unc_th)
            n_au = torch.sum( confidences[n_au_mask] * torch.tanh(unc[n_au_mask]) )

            n_ic_mask = torch.logical_and(labels != predictions, unc <= unc_th)
            n_ic = torch.sum( (1 - confidences[n_ic_mask] ) * (1 - torch.tanh(unc[n_ic_mask]) ) )

            n_iu_mask = torch.logical_and(labels != predictions, unc > unc_th)
            n_iu = torch.sum( (1 - confidences[n_iu_mask] ) *  torch.tanh(unc[n_iu_mask]) )

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            avu_list.append(AvU)
            unc_list.append(unc_th)

        #auc_avu = auc(th_list, avu_list)
        auc_avu = my_auc(th_list, torch.stack(avu_list))
        CE_loss = -torch.sum(y * torch.log(probs.clamp(min=self.eps)), dim=1).mean()

        focal_loss, CE_loss = self.focal(probs, y, weights)
        avu_loss = -self.beta * torch.log(auc_avu + self.eps) + focal_loss
        #pdb.set_trace()
        return avu_loss, CE_loss

class SoftAUAvULoss(nn.Module):
    """
    Calculates Soft Accuracy vs Uncertainty Loss of a model.
    The input to this loss is probabilites from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]

    The reference codes are from: 
    1.) https://github.com/IntelLabs/AVUC/blob/main/src/avuc_loss.py
    2.) https://github.com/google/uncertainty-baselines/blob/main/experimental/caltrain/secondary_losses.py
    """
    def __init__(self, beta=1, num_classes=10):
        super(SoftAUAvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10
        self.entmax = torch.log(torch.tensor(num_classes) ).to(device)
        self.focal = FocalLoss(gamma=1)

    def entropy(self, prob):
        return -torch.sum(prob * torch.log(prob.clamp(self.eps)), dim=1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def soft_T(self, e, temp=0.01, theta=0.1):
        numerator = e * (1 - theta)
        denominator = (1 - e) * theta
        frac = numerator/denominator.clamp(self.eps)
        v = 1/temp * torch.log(frac.clamp(self.eps))
        #print(v.mean())
        return torch.sigmoid(v)

    def forward(self, probs, y, weights, type=0):

        confidences, predictions = torch.max(probs, 1)
        labels = torch.argmax(y)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        th_list = torch.linspace(0, 1, 21, requires_grad=True).to(device)
        umin, umax = torch.min(unc), torch.max(unc)

        avu_list = []
        unc_list = []

        #auc_avu = torch.ones(1, device=labels.device, requires_grad=True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t) * (umax - umin))
            
            n_ac = torch.zeros(1, device=labels.device)
            n_ic = torch.zeros(1, device=labels.device)
            n_au = torch.zeros(1, device=labels.device)
            n_iu = torch.zeros(1, device=labels.device)
            
            #Use masks and logic operators to compute the 4 differentiable proxies
            n_ac_mask = torch.logical_and(labels == predictions, unc <= unc_th)
            n_ac = torch.sum( (1 - self.soft_T(unc[n_ac_mask]/self.entmax)) * (1 - torch.tanh(unc[n_ac_mask]) ) )

            n_au_mask = torch.logical_and(labels == predictions, unc > unc_th)
            n_au = torch.sum( self.soft_T(unc[n_au_mask]/self.entmax) * torch.tanh(unc[n_au_mask]) )

            n_ic_mask = torch.logical_and(labels != predictions, unc <= unc_th)
            n_ic = torch.sum( (1 - self.soft_T(unc[n_ic_mask]/self.entmax) ) * (1 - torch.tanh(unc[n_ic_mask]) ) )

            n_iu_mask = torch.logical_and(labels != predictions, unc > unc_th)
            n_iu = torch.sum( self.soft_T(unc[n_iu_mask]/self.entmax) *  torch.tanh(unc[n_iu_mask]) )

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            avu_list.append(AvU)
            unc_list.append(unc_th)

        #pdb.set_trace()
        #auc_avu = auc(th_list, avu_list)
        auc_avu = my_auc(th_list, torch.stack(avu_list))
        focal_loss, CE_loss = self.focal(probs, y, weights)

        Savu_loss = -self.beta * torch.log(auc_avu.clamp(min=self.eps)) + focal_loss        
        return Savu_loss, CE_loss

class CPCLoss(nn.Module):
    #Code from: https://github.com/by-liu/CALS/blob/main/calibrate/losses/cpc_loss.py
    def __init__(self, lambd_bdc=1.0, lambd_bec=1.0, ignore_index=-100, num_classes=10):
        super().__init__()
        self.lambd_bdc = lambd_bdc
        self.lambd_bec = lambd_bec
        self.ignore_index = ignore_index
        self.eps = 1e-7
        self.num_classes = num_classes
        self.cross_entropy = FocalLoss(gamma=0)

    @property
    def names(self):
        return "loss", "loss_ce", "loss_bdc", "loss_bec"

    def bdc(self, logits, targets_one_hot):
        # 1v1 Binary Discrimination Constraints (BDC)
        logits_y = logits[targets_one_hot == 1].view(logits.size(0), -1)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        loss_bdc = - F.logsigmoid(logits_y - logits_rest).sum() / (logits.size(1) - 1) / logits.size(0)

        return loss_bdc

    def bec(self, logits, targets_one_hot):
        # Binary Exclusion Constraints (BEC)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        diff = logits_rest.unsqueeze(2) - logits_rest.unsqueeze(1)
        loss_bec = - torch.sum(
            0.5 * F.logsigmoid(diff + self.eps)
            / (logits.size(1) - 1) / (logits.size(1) - 2) / logits.size(0)
        )

        return loss_bec

    def forward(self, inputs, targets_one_hot, weights):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            
            targets = targets_one_hot.argmax(1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        focal_loss, loss_ce = self.cross_entropy(inputs, targets_one_hot, weights)
        loss_bdc = self.bdc(inputs, targets_one_hot)
        loss_bec = self.bec(inputs, targets_one_hot)

        CPCloss = loss_ce + self.lambd_bdc * loss_bdc + self.lambd_bec * loss_bec

        return CPCloss, loss_ce

### FOR DMUE ###

class SP_KD_Loss(object):
    def __call__(self, G_matrixs, G_main_matrixs):
        # G_matrixs: List
        # G_main_matrixs: List
        G_err = [F.mse_loss(G_aux, G_main) for G_aux, G_main in zip(G_matrixs, G_main_matrixs)]
        G_err = sum(G_err) / len(G_main_matrixs)

        return G_err

class SoftLoss(object):
    def __call__(self, outputs_x, targets_x, softlabel_x, epoch=None):
        # output_x: tensor
        # softlabel_x: tensor , size like output_x
        probs_x = torch.softmax(outputs_x, dim=1).cuda()
        mask = torch.ones_like(probs_x).scatter_(1, targets_x.view(-1, 1).long(), 0).cuda()
        probs_x = probs_x * mask
        Lsoft = torch.mean((probs_x - softlabel_x)**2)
        return Lsoft