import torch
import torch.nn as nn
import torch.nn.functional as F

from Params import Params

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernels = torch.tensor( [[[[-1,1],[0,0]], [[-1,0],[1,0]]]], dtype=torch.float )
        self.kernels = self.kernels.to(Params.Device)

    def forward(self, flow):
        grad_maps = F.conv2d(flow, self.kernels, padding='same')
        l2_norm = torch.linalg.matrix_norm(grad_maps)
        return l2_norms.sum()
