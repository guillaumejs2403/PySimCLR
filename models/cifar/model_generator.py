import torch
import torch.nn as nn
import torch.nn.functional as F

import models.cifar.resnet as resnet


def get_model(model):
    if 'resnet' in model:
        return resnet.ResNet(model)
    else:
        raise ValueError(f'{model} not implemented.')


class SimCLR_Model(nn.Module):
    def __init__(self, config_m):
        super().__init__()
        self.backbone = get_model(config_m['backbone'])
        ft_dim = self.backbone.output_dim
        # self.projection = nn.Sequential(nn.Linear(ft_dim, ft_dim, bias=True),
        #                                 nn.BatchNorm2d(ft_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(ft_dim, model['model']['backbone'], bias=False))
        self.projection = nn.Sequential(nn.Linear(ft_dim, ft_dim, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(ft_dim, config_m['out_dim'], bias=False))
        
    def forward(self, h):
        h = self.backbone(h)
        h = torch.flatten(h, 1)

        z = self.projection(h)
        z = F.normalize(z, dim=1)
        return h, z
