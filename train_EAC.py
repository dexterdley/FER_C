#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:30:17 2020
"""
# imports
import torch, os
torch.cuda.set_device(0) # set to cuda:1
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
from FER_dataloader import FER_dataset
from PIL import Image
from utils import *
import pdb
import argparse,random
from torch.utils.tensorboard import SummaryWriter
from mek_models import SwinTransformer
from losses import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/AffectNet/data/Manually_Annotated/Manually_Annotated_Images/', help='AffectNet dataset path.')
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
    parser.add_argument('--relabel_epoch', default=2, type=int, help='Start relabel epoch')
    parser.add_argument("--mbls", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Apply MBLS algorithm")
    parser.add_argument("--seed", type=int, default=0, 
        help="seed of the experiment")
    parser.add_argument('--gamma', type=float, default=1.0, help='Focal factor')
    parser.add_argument('--constraints', type=int, default=0, help='Max Ent mode constraints 1-Mu 2-Variance 3-Poly')

    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="FER-C Benchmark Seed1",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    return parser.parse_args()


class Res18Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=8, swin_num_features=512, drop_rate=0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        
        resnet  = torchvision.models.resnet18(pretrained)
        self.model = nn.Sequential(*list(resnet.children())[:-2]) # after avgpool 512x1

        self.fc_in_dim = swin_num_features
        self.fc1 = nn.Sequential(nn.Linear(self.fc_in_dim, 2) ,nn.Tanh())
        self.fc2 = nn.Linear(self.fc_in_dim, num_classes) # new fc layer 512x8

        #attention consistency
        self.norm = nn.LayerNorm(self.fc_in_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        
        x = self.model(x)
        x = x.view(x.size(0), x.size(2)* x.size(3), x.size(1))
        x = self.norm(x)  # B L C

        feature = self.avgpool(x.transpose(1, 2))  # B C 1
        feature = torch.flatten(feature, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(feature)
        
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)

        fc_weights = self.fc2.weight
        fc_weights = fc_weights.view(1, self.num_classes, self.fc_in_dim, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad = False)

        # attention
        B, L, C = x.shape
        x = F.interpolate(x.unsqueeze(1), size=(self.num_classes**2, self.fc_in_dim), mode='nearest') #upsample for general compatibility to all num_classes
        x = x.squeeze(1)
        feat = x.transpose(1, 2).view(B, 1, C, self.num_classes, self.num_classes) # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W
        return hm, x1, x2 #regression output, classification output, attention map

def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip, reduction='none')
    return flip_loss_l    


def generate_flip_grid(w, h):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

def run_training():

    args = parse_args()
    batch_size = args.batch_size
    root = args.file_path
    num_epochs = args.epochs
    save_interval = args.save_intervals

    if args.dataset == 'affectnet':
        num_classes = 8
        train_df = pd.read_csv('./AffectNet/labels/affectnet_train.csv')
        test_df = pd.read_csv('./AffectNet/labels/affectnet_validation.csv') #validation set as test set

    elif args.dataset == 'affwild':
        num_classes = 7
        train_df = pd.read_csv('./Affwild/labels/affwild_training.csv')
        test_df = pd.read_csv('./Affwild/labels/affwild_validation.csv')

    elif args.dataset == 'rafdb':
        num_classes = 6
        train_df = pd.read_csv('./RAF-DB/labels/mega_training.csv')
        test_df = pd.read_csv('./RAF-DB/labels/mega_testing.csv')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Res18Feature(pretrained=True, drop_rate=args.drop_rate, num_classes=num_classes)        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    val_transformations = transforms.Compose([
                transforms.Resize(size=(224, 224),interpolation=2),
                transforms.ToTensor(), #3*H*W, [0, 1]
                normalize, # normalize with mean/std
                transforms.RandomErasing(scale=(0.02, 0.25))])

    if args.dataset == 'affwild':
        val_df = train_df.sample(n=2000)
    else:
        val_df = pd.concat([train_df[train_df['expression'] == k].sample(n=250, random_state=0) for k in range(num_classes)])

    train_df = train_df.loc[~train_df.index.isin(val_df.index)] # remainder for training
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    count, class_weights = get_weights(train_df, num_classes)
    class_weights = torch.tensor(class_weights).to(device)
    ratio = torch.ones(num_classes) * 1/num_classes
    if args.dataset == 'affwild':
        class_weights = None

    if args.noise_ratio > 0:
        print('Adding %.2f noise' %args.noise_ratio )
        train_df = add_noise(train_df, args.noise_ratio, num_classes)

    train_dataset = FER_dataset(root, train_df, dataset=args.dataset, transform=val_transformations)
    val_dataset = FER_dataset(root, val_df, dataset=args.dataset, transform=val_transformations)
    test_dataset = FER_dataset(root, test_df, dataset=args.dataset, transform=val_transformations)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size*2, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size*2, num_workers=args.workers, pin_memory=True)

    if args.constraints == 0:
        criterion = FocalLoss(gamma=0) #Regular CE loss

    elif args.constraints == 1:
        criterion = FocalLoss(gamma=args.gamma)

    elif args.constraints == 2:
        criterion = InvFocalLoss(gamma=args.gamma)

    elif args.constraints == 3:
        criterion = AUAvULoss(beta=3.0)

    elif args.constraints == 4:
        criterion = SoftAUAvULoss(num_classes=num_classes)

    elif args.constraints == 5:
        criterion = PolyLoss()
    
    elif args.constraints == 6:
        criterion = MaxEntLoss(ratio=ratio, constraints=args.constraints, num_classes=num_classes, gamma=args.gamma)

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
    
    if args.mbls:
        print('Applying MBLS')
        m = torch.tensor(8).to(device)

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
        CE_losses = []

        num_correct = 0
        num_samples = 0
        num_relabelled = 0

        for batch_idx, (data, GT_classes, GMM_labels) in tqdm(enumerate(train_loader)):
            
            # Get data to cuda if possible
            data = data.to(device) #images
            data_flip = transforms.RandomHorizontalFlip(p=1)(data)
            GT_classes = GT_classes.to(device)
            GMM_labels = GMM_labels.to(device)
            batch_sz = data.size(0)
            

            onehot = convert_to_onehot(GT_classes, num_classes).to(device) #convert classes to onehot representation
            beta = args.fuse_factor

            if args.fuse_type == 'GMM':
                targets_var = beta * onehot + (1 - beta) * GMM_labels

            # forward
            attn_map, _, logits = model(data)
            attn_map1, _, logits1 = model(data_flip)
            class_pred = F.softmax(logits, dim=1)

            grid_l = generate_flip_grid(num_classes, num_classes).cuda(non_blocking=True)
            flip_loss = ACLoss(attn_map, attn_map1, grid_l, class_pred)    #N*C*C*C

            ''' Computes the classification loss (Cross entropy loss) '''
            loss, CE_loss = criterion(class_pred, targets_var, class_weights)
            CE_losses.append(CE_loss.item())
            
            loss = loss + 0.1 * flip_loss.mean()

            if args.mbls:
                mbls_loss = torch.relu(logits.max(1)[0].unsqueeze(1) - logits - m ).sum(1)
                loss += mbls_loss.mean()

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
        ce_val_loss, scores, val_acc, val_F1, ECE, MCE, adaece, KSE = evaluate_model(val_loader, model, num_classes)
        
        train_stats = ",\t".join([
                f'Epoch: {epoch} TRAIN Total loss: {sum(losses)/len(losses):.3f}',
                f'Acc: {train_accuracy:.3f}',
                f"CE loss: {sum(CE_losses)/len(CE_losses):.3f}",
                ])

        val_stats = ",\t".join([
                f'CE loss: {ce_val_loss:.3f}',
                f"ECE: {ECE:.3f}",
                f"MCE: {MCE:.3f}",
                f"KSE: {KSE:.3f}",
                ])

        print(train_stats)
        print(val_stats)
        print(scores)

        writer.add_scalar("Train/Accuracy", train_accuracy, epoch)
        writer.add_scalar("Train/CE loss", sum(CE_losses)/len(CE_losses), epoch)
        
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Val/F1", val_F1, epoch)
        writer.add_scalar("Val/ECE", ECE, epoch)
        writer.add_scalar("Val/MCE", MCE, epoch)
        
        if val_F1 > best_F1 and epoch != 0:     
            checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, 'best', expt_name)
            best_F1 = val_F1

    #evaluate the best performing model
    best_model_path = './models/' + expt_name + '/' + 'best_' + expt_name + '.pth.tar'
    best_model = load_pretrained(model, best_model_path)
    
    ce_test_loss, scores, test_acc, test_F1, test_ECE, test_MCE, test_adaece, test_KSE = evaluate_model(test_loader, best_model, num_classes)
    # test on test set
    test_stats = ",\t".join([
                f'Test Acc: {test_acc:.3f}',
                f'Test F1: {test_F1:.3f}',
                f"ECE: {test_ECE:.3f}",
                f"MCE: {test_MCE:.3f}",
                f"NLL: {ce_test_loss:.3f}",
                f"KSE: {test_KSE:.3f}",
                ])
    print(test_stats)

    writer.add_scalar("Test/Accuracy", test_acc, epoch)
    writer.add_scalar("Test/F1", test_F1, epoch)
    writer.add_scalar("Test/ECE", test_ECE, epoch)
    writer.add_scalar("Test/MCE", test_MCE, epoch)
    writer.add_scalar("Test/NLL", ce_test_loss, epoch)
    writer.close()

if __name__ == "__main__":                    
    run_training()