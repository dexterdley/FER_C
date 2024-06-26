import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from resnet_multibranch import ResNet, BasicBlock, Bottleneck
'''
Code reference: https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DMUE
'''

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:  # need to be comment when BiasInCls is True
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=8):
        super(Backbone, self).__init__()
        
        self.num_classes = num_classes
        self.num_branches = self.num_classes + 1
        self.BiasInCls = False
        self.neck = False
        self.KD_whole_cls = False
        self.second_order_statics = 'mean'

        # base model
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(num_classes=self.num_classes,
                        num_branches=self.num_branches,
                        block=BasicBlock,
                        layers=[2, 2, 2, 2])

        # pooling after base
        self.gap = nn.AdaptiveAvgPool2d(1)

        # loss
        self.classifiers = []
        
        for i in range(self.num_branches):
            output_classes = self.num_classes if i is self.num_branches-1 else self.num_classes -1
            self.classifiers.append(nn.Linear(self.in_planes, output_classes, bias=self.BiasInCls))
            self.classifiers[i].apply(weights_init_classifier)
        self.classifiers = nn.ModuleList(self.classifiers)
        
        
        self.sigmoid = nn.Sigmoid()
        input_dim = 2 * self.num_classes
        self.project_w = nn.Sequential(nn.Linear(input_dim, self.num_classes),nn.PReLU(), nn.Linear(self.num_classes, 1))
            
        if self.neck:
            # print('Using bnneck')
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, training_phase=None, c=None):
        x_final, x_list, softlabel_list, targets, inds, inds_softlabel = self.base(x, label, training_phase=training_phase, c=c)
        if training_phase == 'normal':
            x_final = self.gap(x_final).squeeze(2).squeeze(2)
            x_final = self.bottleneck(x_final) if self.neck else x_final
            x_final = self.classifiers[self.num_branches - 1](x_final)
                              
            return x_final, x_list, targets, None

        # train auxiliary & main branch
        x_list = [self.gap(_x).squeeze(2).squeeze(2) for _x in x_list]
        x_list = [self.bottleneck(_x) for _x in x_list] if self.neck else x_list
        x_list = [self.classifiers[i](x_list[i]) for i in range(self.num_branches - 1)]
        x_final = self.gap(x_final).squeeze(2).squeeze(2)
        
        # before dropout cal cos_simlarity in batch
        if training_phase == 'sp_confidence':
            cos_dot_product_matrix = x_final.mm(x_final.t())
            x_final_norm = torch.norm(x_final, dim=1).unsqueeze(1)
            x_norm_product = x_final_norm.mm(x_final_norm.t())
            cos_similarity = cos_dot_product_matrix / x_norm_product
            
        x_final = self.bottleneck(x_final) if self.neck else x_final

        # make distribution
        softlabel_list = [self.gap(_x).squeeze(2).squeeze(2) for _x in softlabel_list]
        softlabel_list = [self.bottleneck(_x) for _x in softlabel_list] if self.neck else softlabel_list

        if training_phase == 'sp_confidence':
            # sp kd
            G_matrixs = [fm.mm(fm.t()) for fm in x_list]
            G_main = x_final.mm(x_final.t())
            # construct sub-matrix of G_main before l2-norm
            G_main_matrixs = [G_main[ind[:,0], :][:, ind[:,0]] for ind in inds]
            # l2 norm 
            G_matrixs = [F.normalize(G, p=2, dim=1) for G in G_matrixs]
            G_main_matrixs = [F.normalize(G, p=2, dim=1) for G in G_main_matrixs]

            softlabel_list = [self.classifiers[i](softlabel_list[i]) for i in range(self.num_branches - 1)]
            x_final = self.classifiers[self.num_branches - 1](x_final)    
            
            # convert label into one-hot vector 
            label = label.unsqueeze(1)
            one_hot = torch.zeros(label.shape[0], self.num_classes).cuda().scatter_(1, label, 1)
            # cal mean according to different classes
            cos_matrixs = [cos_similarity[:, ind[:, 0]] for ind in inds_softlabel] # may be change by torch.Tensor.indexadd(dim, index, tensor)
            
            #cos_matrixs_mean = [torch.mean(cos_m, dim=1, keepdim=True) for cos_m in cos_matrixs] #bug arises if minority class does not appear in mini-batch
            cos_matrixs_mean = [torch.mean(cos_m, dim=1, keepdim=True) if cos_m.size(1) > 0 else torch.zeros(cos_m.size(0), 1).cuda() for cos_m in cos_matrixs]


            cos_mean = torch.cat(tuple(cos_matrixs_mean), dim=1)# bs x num_class
            # scale to 0-1, the effect is minor
            cos_mean = (cos_mean - cos_mean.min(dim=1, keepdim=True)[0]) / (cos_mean.max(dim=1, keepdim=True)[0] - cos_mean.min(dim=1, keepdim=True)[0])
            statics = torch.cat((cos_mean, one_hot), dim=1)

            score = self.project_w(statics)
            score = self.sigmoid(score)
            atten_x_final = x_final * score

            return x_final, x_list, targets, softlabel_list, G_matrixs, G_main_matrixs, score, atten_x_final
        
    def load_param(self):
        pretrained = "out_dir_res18/mv_epoch_17.pt"
        param_dict = torch.load(pretrained, map_location=lambda storage,loc: storage.cpu())
        print('Pretrained choice ', pretrained)

        for i in param_dict['state_dict']:
            if ('loss_layer' in i) or ('fc' in i):
                continue
            else:
                layer_name = i.split('.')[1]
                if 'layer4' == layer_name:
                    op_name = i[16:]
                    for branch_idx in range(self.num_branches):
                        j = 'base.'+layer_name +'_'+str(branch_idx)+'.'+op_name
                        assert j in self.state_dict()
                        self.state_dict()[j].copy_(param_dict['state_dict'][i])
                else:
                    j = 'base'+i.split('feat_net')[1]
                    assert j in self.state_dict()   
                    self.state_dict()[j].copy_(param_dict['state_dict'][i])