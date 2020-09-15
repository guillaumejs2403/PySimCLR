import torch
import torch.nn as nn

import core.losses as losses


class simclr_step(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.criterion = losses.InfoNCELoss(temperature, device)
        self.lin_criterion = nn.CrossEntropyLoss()

    def forward(self, phase='unsupervised', **kwargs):

        if phase == 'unsupervised':
            return self.unsupervised_step(**kwargs)

        elif phase == 'linear':
            return self.linear_trainig_step(**kwargs)

        elif phase == 'linear test':
            return self.linear_test_step(**kwargs)

    def unsupervised_step(self, im1, im2, model):
        _, z1 = model(im1)
        _, z2 = model(im2)

        loss = self.criterion(z1, z2)
        return loss

    def linear_trainig_step(self, img, lbl, model, linear):
        with torch.no_grad():
            h, _ = model(img)
        h = linear(h)
        loss = self.lin_criterion(h, lbl)

        top1 = (h.argmax(dim=1) == lbl).sum().item()

        return loss, top1

    def linear_test_step(self, img, lbl, model, linear):
        with torch.no_grad():
            h, _ = model(img)
            h = linear(h)
            loss = self.lin_criterion(h, lbl)

        top1 = (h.argmax(dim=1) == lbl).sum().item()

        return loss, top1
