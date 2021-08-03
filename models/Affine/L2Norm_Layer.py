import torch
import torch.nn as nn

class L2Norm_Layer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.eps = 1e-2
        self.dim = dim

    def forward(self, x):
        massp = torch.sum(torch.square(x), dim=self.dim) + self.eps
        return x * torch.rsqrt(massp)
