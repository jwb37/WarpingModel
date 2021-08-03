import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Translation.Correlation import correlation_map

from Params import Params


class WST_Loss(nn.Module):
    def __init__(self, visualizer=None):
        super().__init__()
        self.visualizer = visualizer


    def forward(self, featsA, warpedfeatsA, featsB, warp_grid, patch_size=9):
        corr = correlation_map(warpedfeatsA, featsB, flatten=True)

        # N x P_B x P_A where P_A, P_B are the number of pixels in 
        flat_corr = corr.flatten(start_dim=2)
        N, P_B, P_A = flat_corr.size()
        # We wish each pixel in warped A to be most like the matching pixel B out of all the pixels in B
        target = torch.arange(P_B, dtype=torch.long).to(Params.Device)
        target = target.view(1, P_A)
        target = target.repeat(N, 1)

        loss = F.cross_entropy(flat_corr, target)

        return loss
