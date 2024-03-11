#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:30:17 2020

@author: aisg
"""
# imports
import torch, os
torch.cuda.set_device(0) 
print('Current device:', torch.cuda.current_device()) 

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import copy

import pandas as pd
import torchvision

import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from affectnet_dataloader import AffectNet_dataset
from PIL import Image
from utils import *
import pdb
import argparse,random
from torch.utils.tensorboard import SummaryWriter
from losses import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

hdd_root = '/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/' 
if os.path.exists(hdd_root) == False:
    hdd_root = '/mnt/hdd/dexter_storage'


''' 
Helper functions, to be moved to utils.py in future
'''
def filter_using_std(x, mean, std, max_deviations):
        
    distance_from_mean = abs(x - mean) # MAE from class mean
        
    valid_pts = distance_from_mean < max_deviations * std
    
    #pdb.set_trace()
    return valid_pts 

def compute_statistics(array):
        
        min_value = np.min(array)
        max_value = np.max(array)
        mean = np.mean(array) #average
        std = np.std(array) #std deviation
        
        return min_value, max_value, mean, std

def prepare_df(dataframe):
    '''
    Accepts affectnet dataframe from, returns dataframe without,
    none, uncertain and no-face
    
    Missing_images contains the images lost during downloading/decompression
    
    '''
    good_idx = np.where(dataframe['expression'] < 8)[0] #filter out none, uncertain no-face
    dataframe = dataframe.iloc[good_idx].reset_index()
    
    missing_images = ['1066/850128674e45b5713f195e45ae85cd11aadbf08c552c660422271ea9.jpg',
                      '461/d45ace15e1ed59b21e7665fe337d056d7236bf1b668a0b64cdaeea12.jpg',
                      '720/2f71e90c66de7c0b59a2421d652d2241186b7777d9b7acf63a3bb32a.jpg',
                      '414/1d8c0d6ef197634bb27c26d895627b67301e0b3d5f4559057b7ed108.jpg',
                      '901/733447fbdd56df6a308f1c6bf223b4d4556bee059631536db27c86ca.jpg',
                      '674/e452d8455e7fe35a04edd54afb19baf6867a357489998d8e90530080.jpg' ]
    
    for item in missing_images:
        
        idx = np.where(dataframe['subDirectory_filePath'] == item)[0]
        
        if len(idx) > 0:
            #print(idx)    
            dataframe = dataframe.drop(idx.item()).reset_index(drop=True)
            
            
    return dataframe
    
def get_centroids(df, num_classes, max_deviations):
    '''
    Accepts train_df and compute the mean and std of the valence and arousal values
    return a dict of centroids of points within a specified std deviation
    '''
    
    classes = np.arange(num_classes)

    centroids = {}

    for k in classes:
        class_idx = np.where(df['expression'] == k)[0]
    
        min_valence, max_valence, avg_valence, std_valence = compute_statistics(df['valence'][class_idx])
        min_arousal, max_arousal, avg_arousal, std_arousal = compute_statistics(df['arousal'][class_idx])
    
        x = df['valence'][class_idx]
        y = df['arousal'][class_idx]
    
        x_idx = filter_using_std(x, avg_valence, std_valence, max_deviations)
        y_idx = filter_using_std(y, avg_arousal, std_arousal, max_deviations) 
        idx = np.logical_and(x_idx, y_idx)
    
        #recompute mean
        avg_valence = np.mean(x[idx])
        avg_arousal = np.mean(y[idx])
   
        #print('Class centroid: %d' %k, (avg_valence,avg_arousal) ,'Class std:', (std_valence, std_arousal) )
        centroids[k] = np.array([avg_valence, avg_arousal])
    
    return centroids

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
    
    #ratio = class_weights/ np.sum(class_weights)
    #pdb.set_trace()
    return class_weights, max_class/class_weights #.to(device)

def get_nearest_neighbour(predictions, centroids):
    
    distances = []
    
    for k in centroids:
        
        dist = np.sqrt(np.square(predictions[0] - centroids[k][0]) + np.square(predictions[1] - centroids[k][1]))
        distances.append(dist.item())
    
    return np.argmin(distances), distances

def weighted_mse(A, B, class_weights):
    y = torch.square(A - B)
    
    y = y * class_weights
    
    out = torch.mean(y)
    return out

def weighted_negative_log_likelihood(pred, y, class_weights):

    out = torch.log(pred) * y
    
    out = -torch.sum(out, dim=1)  * class_weights #changed 
    
    #pdb.set_trace()
    return torch.mean(out)

def compute_losses(pred, target, class_pred, targets_var, GT_labels, class_weights):
    '''
    Computes the regression loss (L2 loss) and classification loss (Cross entropy loss)
    between predictions and target samples
    '''
    
    #Normal weights
    weights = torch.zeros(targets_var.shape) # efficiently assign class weights
    h_weights = torch.ones(targets_var.shape)
    
    onehot = convert_to_onehot(GT_labels)
    idx = np.where(onehot.cpu() ==1)

    weights[idx] = torch.tensor(class_weights[idx[1]]).float()
    
    #compute weighted MSE loss
    mse_weights = torch.max(weights, dim=1)[0].to(device)
    w_v_loss = weighted_mse(pred[:,0], target[:,0], mse_weights)
    w_a_loss = weighted_mse(pred[:,1], target[:,1], mse_weights)
    
    R_loss = w_v_loss + w_a_loss 
    
    CE_loss = weighted_negative_log_likelihood(class_pred, targets_var, mse_weights)
    cos_loss = cosine_loss(class_pred, targets_var, mse_weights)
    
    return R_loss, CE_loss, cos_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/AffectNet/data/Manually_Annotated/Manually_Annotated_Images/', 
        help='AffectNet dataset path.')
    parser.add_argument('--dataset', type=str, default='affectnet', help='Train on AffectNet')
    parser.add_argument('--expt_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--fuse_factor', type=float, default=1.0, help='Fuse factor')
    parser.add_argument('--fuse_type', type=str, default='GMM', help='Fusion type, GMM or VA')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Initial learning rate for sgd.')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='Synthetic noisy label ratio')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--save_intervals', type=int, default=10, help='Num of save intervals')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    parser.add_argument("--mbls", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,help="Apply MBLS algorithm")
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument('--gamma', type=float, default=1.0, help='Focal factor')
    parser.add_argument('--constraints', type=int, default=0, help='Max Ent mode constraints 1-Mu 2-Variance 3-Poly')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="FER-C Benchmark Seed2",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    return parser.parse_args()

class Res18Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=8, drop_rate=0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet  = torchvision.models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
        
        self.fc1 = nn.Sequential(nn.Linear(fc_in_dim, 2) ,nn.Tanh())
        self.fc2 = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x8

    def forward(self, x):
        x = self.features(x)
        
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
     
        x1 = self.fc1(x)
        x2 = self.fc2(x) #usage

        #pdb.set_trace()
        return x, x1, x2 # regression output, classification output

def run_training():

    args = parse_args()
    batch_size = args.batch_size
    root = args.file_path
    num_epochs = args.epochs
    save_interval = args.save_intervals

    if args.dataset == 'affectnet':
        num_classes = 8
        train_df = pd.read_csv(hdd_root + '/AffectNet/labels/affectnet_train.csv')
        test_df = pd.read_csv(hdd_root + '/AffectNet/labels/affectnet_validation.csv') #validation set as test set

    elif args.dataset == 'affwild':
        num_classes = 7
        train_df = pd.read_csv(hdd_root + '/Affwild/affwild_training.csv')
        test_df = pd.read_csv(hdd_root + '/Affwild/affwild_validation.csv')

    elif args.dataset == 'rafdb':
        num_classes = 6
        train_df = pd.read_csv(hdd_root + '/RAF-DB/mega_training.csv')
        test_df = pd.read_csv(hdd_root + '/RAF-DB/mega_testing.csv')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Res18Feature(pretrained=True, drop_rate=args.drop_rate, num_classes=num_classes)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=30,translate=(.1, .1),scale=(1.0, 1.25),resample=Image.BILINEAR),
                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                transforms.Resize(size=(224,224),interpolation=2),
                transforms.ToTensor(), #3*H*W, [0, 1]
                normalize, # normalize with mean/std
                transforms.RandomErasing(scale=(0.02,0.25))])


    val_transformations = transforms.Compose([
                transforms.Resize(size=(224,224),interpolation=2),
                transforms.ToTensor(), #3*H*W, [0, 1]
                normalize]) # normalize with mean/std

    if args.dataset == 'affwild':
        val_df = train_df.sample(n=2000)
    else:
        val_df = pd.concat([train_df[train_df['expression'] == k].sample(n=250, random_state=0) for k in range(num_classes)])

    train_df = train_df.loc[~train_df.index.isin(val_df.index)] # remainder for training
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    count, class_weights = get_weights(train_df, num_classes)
    class_weights = torch.tensor(class_weights)

    centroids = get_centroids(train_df, num_classes, max_deviations=1)
    ratio = torch.ones(num_classes) * 1/num_classes
    if args.dataset == 'affwild':
        class_weights = None

    if args.noise_ratio > 0:
        print('Adding %.2f noise' %args.noise_ratio )

        train_df = add_noise(train_df, args.noise_ratio, num_classes)

    train_dataset = AffectNet_dataset(root, train_df, transform=transformations)
    val_dataset = AffectNet_dataset(root, val_df, transform=val_transformations)
    test_dataset = AffectNet_dataset(root, test_df, transform=val_transformations)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)

    if args.constraints == 0:
        criterion = FocalLoss(gamma=0) #Regular CE loss

    
    #Loss and optimiser
    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-5)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-5)                           
    else:
        raise ValueError("Optimizer not supported.")

    model.to(device)

    # Train the network
    expt_name = str(args.seed) + '_' + args.dataset + '_' + args.expt_name
    print('Running:', expt_name, 'Batch size:', batch_size)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=expt_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{expt_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    best_F1 = 0
   

    for epoch in range(num_epochs):
        model.train()
        
        losses = []
        R_losses = []
        CE_losses = []
        cos_losses = []

        num_correct = 0
        num_samples = 0
    
        for batch_idx, (data, targets, GT_classes, GMM_labels) in tqdm(enumerate(train_loader)):
            
            # Get data to cuda if possible
            data = data.to(device) #images
            targets = targets.to(device) #valence, arousal
            GT_classes = GT_classes.to(device)
            GMM_labels = GMM_labels.to(device)


            onehot = convert_to_onehot(GT_classes, num_classes).to(device) #convert classes to onehot representation
            beta = args.fuse_factor

            if args.fuse_type == 'GMM':
                targets_var = beta * onehot + (1 - beta) * GMM_labels

            elif args.fuse_type == 'VA':
                va_onehot = get_va_labels(centroids, targets).to(device)
                targets_var = beta * onehot + (1 - beta) * va_onehot

            # forward
            _, pred, logits = model(data)
            class_pred = F.softmax(logits, dim=1)

            ''' Computes the classification loss (Cross entropy loss) '''
            R_loss, CE_loss, cos_loss = compute_losses(pred, targets, class_pred, targets_var, GT_classes, class_weights)
            
            loss = R_loss + CE_loss + cos_loss

            R_losses.append(R_loss.item())
            CE_losses.append(CE_loss.item())
            cos_losses.append(cos_loss.item())

            # backward
            optimizer.zero_grad()
            losses.append(loss.item())
            loss.backward()
        
            # gradient descent or adam step
            optimizer.step()

            num_correct += torch.sum(torch.eq(class_pred.argmax(1), GT_classes)).item()
            num_samples += class_pred.size(0)


        print('Epoch {}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))

        train_accuracy = float(num_correct)/float(num_samples)
        mse_val_loss, ce_val_loss, cos_val_loss, rmse_v, rmse_a, CCC_V, CCC_A, corr_V, corr_A, scores, val_acc, val_F1, ECE, MCE, NLL, brier = evaluate_GMM(val_loader, model, num_classes)
        
        train_stats = ",\t".join([
                f'Epoch: {epoch} TRAIN Total loss: {sum(losses)/len(losses):.3f}',
                f'Acc: {train_accuracy:.3f}',
                f"R loss: {sum(R_losses)/len(R_losses):.3f}",
                f"CE loss: {sum(CE_losses)/len(CE_losses):.3f}",
                f"Cos loss: {sum(cos_losses)/len(cos_losses):.3f}",
                ])

        val_stats = ",\t".join([
                f'CE loss: {ce_val_loss:.3f}',
                f"val_F1: {val_F1:.3f}",
                f"rmse_V: {rmse_v:.3f}",
                f"rmse_A: {rmse_a:.3f}",
                f"CCC_V: {CCC_V:.3f}",
                f"CCC_A: {CCC_A:.3f}",
                f"corr_V loss: {corr_V:.3f}",
                f"corr_A loss: {corr_A:.3f}",
                ])

        print(train_stats)
        print(val_stats)
        print(scores)

        writer = SummaryWriter("runs/" + expt_name)
        writer.add_scalar('Train R_loss', sum(R_losses)/len(R_losses), epoch)
        writer.add_scalar('Train CE_loss', sum(CE_losses)/len(CE_losses), epoch)
        writer.add_scalar('Train cos_loss', sum(cos_losses)/len(cos_losses), epoch)

        
        if val_F1 > best_F1 and epoch != 0:     
            checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, 'best', expt_name)
            best_F1 = val_F1

    #evaluate the best performing model
    best_model_path = './models/' + expt_name + '/' + 'best_' + expt_name + '.pth.tar'
    best_model = load_pretrained(model, best_model_path)
    
    mse_test_loss, ce_test_loss, cos_test_loss, rmse_v, rmse_a, CCC_V, CCC_A, corr_V, corr_A, scores, test_acc, test_F1, test_ECE, test_MCE, test_NLL, test_brier = evaluate_GMM(test_loader, best_model, num_classes)
    
    # test on test set
    test_stats = ",\t".join([
                f'TEST R loss: {mse_test_loss:.3f}',
                f'CE loss: {ce_test_loss:.3f}',
                f'cos loss: {cos_test_loss:.3f}',
                f"rmse_V: {rmse_v:.3f}",
                f"rmse_A: {rmse_a:.3f}",
                f"CCC_V: {CCC_V:.3f}",
                f"CCC_A: {CCC_A:.3f}",
                f"corr_V loss: {corr_V:.3f}",
                f"corr_A loss: {corr_A:.3f}",
                ])
    print(test_stats)

    writer.add_scalar('Test R_loss', mse_test_loss, epoch)
    writer.add_scalar('Test CE_loss', ce_test_loss, epoch)
    writer.add_scalar('Test cos_loss', cos_test_loss, epoch)
    writer.close()

if __name__ == "__main__":                    
    run_training()