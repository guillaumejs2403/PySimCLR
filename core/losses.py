import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.t = temperature
        self.device = device

    def forward(self, zi, zj):
        B = zi.size(0)
        logits = torch.cat([zi, zj], dim=0)
        logits = torch.mm(logits, logits.t()) / (self.t)
        mask = (torch.ones(2 * B, device=self.device) -
                torch.eye(2 * B, device=self.device))
        mask = mask.bool()
        logits = logits.masked_select(mask).view(2 * B, 2 * B - 1)
        pos = (zi * zj).sum(dim=1) / self.t
        pos = torch.cat([pos, pos])
        loss = -pos + torch.logsumexp(logits, dim=1)
        return loss.mean()
