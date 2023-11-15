# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import random

def mixup_data(x, y, att, use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    att1 = att / (att + att[index])
    att2 = att[index] / (att + att[index])
    mixed_x = att1 * x + att2 * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, att1, att2

def mixup_criterion(criterion, pred, targets_a, targets_b, weights):
    A_loss, A_CE_loss = criterion(pred,targets_a, weights)
    B_loss, B_CE_loss = criterion(pred, targets_b, weights)

    loss = 0.5 * A_loss + 0.5 * B_loss
    CE_loss = 0.5 * A_CE_loss + 0.5 * B_CE_loss
    return loss, CE_loss

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class res18feature(nn.Module):
    def __init__(self, args=None, pretrained=True, num_classes=7, drop_rate=0.0, out_dim=64):
        super(res18feature, self).__init__()

        #'affectnet_baseline/resnet18_msceleb.pth'
        res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], output_dim=1000)
        
        if pretrained:
            msceleb_model = torch.load(args.pretrained_backbone_path)
            state_dict = msceleb_model['state_dict']
            res18.load_state_dict(state_dict, strict=False)

        self.drop_rate = drop_rate
        self.out_dim = out_dim
        self.features = nn.Sequential(*list(res18.children())[:-2])

        self.mu = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5))

        self.log_var = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5))

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, target, phase='train'):

        if phase == 'train':
            x = self.features(x)
            mu = self.mu(x)
            logvar = self.log_var(x)

            mixed_x, y_a, y_b, att1, att2 = mixup_data(mu, target, logvar.exp().mean(dim=1, keepdim=True), use_cuda=True)
            output = self.fc(mixed_x)
            return mixed_x, y_a, y_b, att1, att2, output
        else:
            x = self.features(x)
            output = self.mu(x)
            return self.fc(output)

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x
    

    
class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()
                
        
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2,2,2,2], channels=[64, 128, 256, 512], stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h
    


if __name__ == '__main__':
    res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], output_dim=1000)
    input = torch.randn(1, 3, 224, 224)
    output = res18(input)
    print(output.size())