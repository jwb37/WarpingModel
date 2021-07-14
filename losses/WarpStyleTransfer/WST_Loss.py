import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from Correlation import correlation_map

class WST_Loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, featsA, warpedA, tensorB, warp_grid, patch_size=9):
        corr = correlation_map(warpedA, tensorB, flatten=False)

        half_patch_size = (patch_size-1) // 2

        N, H, W, _, _ = corr.size()
        window = torch.zeros( (H, W), dtype=torch.bool )
        window[:patch_size, :patch_size] = 1

        loss = 0
        patches = torch.zeros(N, patch_size**2)

        iter_range = itertools.product(
            range(half_patch_size, H - half_patch_size),
            range(half_patch_size, W - half_patch_size)
        )

        for i, j in iter_range:
            shifted_window = torch.roll(window, shifts=(i-half_patch_size, j-half_patch_size), dims=(0,1))
            arr = corr[:, i, j, :, :]
            for b in range(N):
                patch = arr[b][shifted_window].flatten()
                loss = loss - F.log_softmax(patch, dim=0).sum()

        return loss
