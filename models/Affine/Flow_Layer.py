import torch
import torch.nn as nn
import torch.nn.functional as F

from Params import Params


class Flow_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tens, flow):
        B, _, H, W = flow.size()
        base_x = torch.linspace(-1,1,W).reshape(1, 1, W, 1).repeat(B,H,1,1)
        base_y = torch.linspace(-1,1,H).reshape(1, H, 1, 1).repeat(B,1,W,1)
        base_grid = torch.cat( (base_x, base_y), dim=-1 ).to(Params.Device)
        warp_grid = base_grid + flow.movedim(1,-1)

        return F.grid_sample(tens, warp_grid, align_corners=False)
