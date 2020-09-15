import torch
import torch.nn as nn

import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        model = getattr(models, resnet)(pretrained=False)

        for (name, m) in model.named_children():
            if name =='conv1':
                m = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                              bias=False)
            elif name == 'maxpool':
                continue
            self.add_module(name, m)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.output_dim = model.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x