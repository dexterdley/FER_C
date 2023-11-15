#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:33:28 2020

@author: aisg
"""
import pandas as pd
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pdb
import ast
from scipy.spatial import cKDTree

class FER_dataset(Dataset): 
    
    def __init__(self, root_path, dataframe, dataset='affectnet', transform=None, SCN=False, LDVLA=False):
        
        self.df = dataframe
        self.transform = transform
        self.root_path = root_path
        self.dataset = dataset
        self.SCN = SCN
        self.LDVLA = LDVLA

        self.image_paths = self.df['subDirectory_filePath'] #image names
        self.expression = self.df['expression'] # class labels

        if 'valence' in self.df.columns:
            self.VA = np.array(self.df[['valence', 'arousal']]) # valence and arousal annotations

        if self.dataset == 'affectnet':
        
            self.x = self.df['face_x'] #pd series bounding box coordinates
            self.y = self.df['face_y']
            self.w = self.df['face_width']
            self.h = self.df['face_height']
        
            self.mixed_labels = self.df['GMM_labels'] #soft gmm labels

        elif self.dataset == 'rafdb':
            self.mixed_labels = self.df['distribution']

        elif self.dataset == 'affwild':
            self.mixed_labels = self.df['GMM_labels']

        if self.LDVLA == True:
            self.tree = cKDTree(self.VA) #Build tree
            self.K = 1 #num neighbors
        
    def __getitem__(self, index):
        
        
        if self.dataset == 'affwild':
            img_path = str(self.df['identity'][index]) + '/' + self.image_paths[index] #for Affwild img path
        else:
            img_path = self.image_paths[index]

        image = Image.open(self.root_path + img_path)

        if self.dataset == 'affectnet':
            x = self.x[index]
            y = self.y[index]
            w = self.w[index]
            h = self.h[index]
            
            bbox = (x, y, w ,h)    
            image = image.crop(bbox) #crop image to given bounding box

        classes = torch.tensor(self.expression[index]) #Class labels
        if type(self.mixed_labels[index]) == str:
            mixed_label = torch.tensor( ast.literal_eval(self.mixed_labels[index]) )
        else:
            mixed_label = torch.tensor( self.mixed_labels[index])

        if self.transform != None:  
            image = self.transform(image)

        if self.LDVLA == True:
            #Compute KNN
            neighbors_index = self.tree.query(self.VA[index], self.K + 1)[1] # finds the 9th nearest neighbors
            
            if index in neighbors_index:
                neighbors_index = neighbors_index[neighbors_index != index] #remove itself
            else:
                neighbors_index = neighbors_index[0:-1] #remove last neighbor

            neighbor_va = self.VA[neighbors_index]  # shape (N,K,2)
            dist = np.sum(np.abs(self.VA[index] - neighbor_va), 1)
            knn_weights = np.exp(-dist / 0.5)

        if self.SCN:
            return [image, classes, mixed_label, index]

        elif self.LDVLA:

            if self.dataset == 'affwild':
                #neighbor_images = torch.stack([ self.transform(Image.open(self.root_path + str(self.df['identity'][neighbors_index].iloc[i] ) + '/' + item)) for i, item in enumerate(self.image_paths[neighbors_index]) ])
                neighbor_images = self.transform(Image.open(self.root_path + str(self.df['identity'][neighbors_index].item() ) + '/' + self.image_paths[neighbors_index].item() )).unsqueeze(0)
            else:
                
                #neighbor_images = torch.stack([ self.transform(Image.open(self.root_path + item)) for item in self.image_paths[neighbors_index] ])
                neighbor_images = self.transform(Image.open(self.root_path + self.image_paths[neighbors_index].item() )).unsqueeze(0)

            return [image, neighbor_images, classes, knn_weights, neighbors_index]
        
        else:
            return [image, classes, mixed_label]
        
    def __len__(self):
        return len(self.expression)