import torch
import torch.nn as nn

from Params import Params

class RTN_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = None
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, corr_vol):
        N, C, H, W = corr_vol.size()
        if self.target is None:
            self.target = torch.ones( (1, H, W), dtype=torch.long ) * ((C - 1)//2)
            self.target = self.target.to(Params.Device)

        target = self.target.repeat(N, 1, 1)
        return self.loss_fn(corr_vol, target)
