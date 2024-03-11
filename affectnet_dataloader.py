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

class AffectNet_dataset(Dataset): 
    
    def __init__(self, root_path , dataframe, seed=0, transform=None, SCN=False):
        
        self.df = dataframe
        
        self.transform = transform
        self.root_path = root_path
        self.image_paths = self.df['subDirectory_filePath'] #image names
        
        self.expression = self.df['expression'] # class labels

        self.valence = self.df['valence'] #annotations
        self.arousal = self.df['arousal'] #annotations
        
        self.x = self.df['face_x'] #pd series bounding box coordinates
        self.y = self.df['face_y']
        self.w = self.df['face_width']
        self.h = self.df['face_height']
        self.seed = seed
        
        self.landmarks = self.df['facial_landmarks']
        self.gmm_labels = self.df['GMM_labels'] #soft gmm labels
        self.SCN = SCN
        

    def __getitem__(self, index):
        
        img_path = self.image_paths[index]
        
        image = Image.open(self.root_path + img_path)
        original_shape = image.size #return width, height of original image
        
        x = self.x[index]
        y = self.y[index]
        w = self.w[index]
        h = self.h[index]
        
        bbox = (x, y, w ,h)
        
        image = image.crop(bbox) #crop image to given bounding box
        classes = torch.tensor(self.expression[index]) #Class labels
        gmm_label = torch.tensor( ast.literal_eval(self.gmm_labels[index]))

        target_v = torch.tensor(self.valence[index])
        target_a = torch.tensor(self.arousal[index]) #labels
        targets = torch.stack([target_v, target_a])

        if self.transform != None:  
            image = self.transform(image)

        if self.SCN:
            return [image, targets, classes, gmm_label, index]
        else:

            return [image, targets, classes, gmm_label]
        
    def __len__(self):
        return len(self.expression)
    
    
    
    
    