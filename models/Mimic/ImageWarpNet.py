import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

from ..BaseModel import BaseModel

from Params import Params
from losses import get_loss_module
from .WarpGenerator import WarpGenerator


class ImageWarpNet(BaseModel):
    def __init__(self, visualizer=None):
        super().__init__(nn.Identity, WarpGenerator, visualizer)

    #-------------------------------------------------------------------------
    # Internals

    def warpTensor(self, tens, flow):
        B, _, H, W = flow.size()
        base_x = torch.linspace(-1,1,W).reshape(1, 1, W, 1).repeat(B,H,1,1)
        base_y = torch.linspace(-1,1,H).reshape(1, H, 1, 1).repeat(B,1,W,1)
        base_grid = torch.cat( (base_x, base_y), dim=-1 ).to(Params.Device)
        warp_grid = base_grid + flow.movedim(1,-1)

        return F.grid_sample(tens, warp_grid, align_corners=False)

    def calc_flow(self, tensorA, tensorB, return_feats=False):
        return self.wg( tensorA )

    #---------------------------------------------------------------
    # Training functions

    def training_step(self, tensorA, tensorB, output_imgs=False):
        self.optimizer.zero_grad()

        flow = self.calc_flow(tensorA, tensorB)
        loss = F.l1_loss(flow, tensorB)
        loss.backward()

        self.optimizer.step()

        if self.visualizer and self.visualizer.save_this_iter:
            self.visualizer.add_tensor( 'imgA', tensorA )

            upscaled_tensorB = F.interpolate( tensorB, scale_factor=8, mode='bilinear', align_corners=False )
            self.visualizer.add_tensor( 'flowReal', upscaled_tensorB )

            upscaled_flow = F.interpolate( flow, scale_factor=8, mode='bilinear', align_corners=False )
            warpedA = self.warpTensor( tensorA, upscaled_flow )
            self.visualizer.add_tensor( 'flowFake', upscaled_flow )
            self.visualizer.add_tensor( 'warpedA', warpedA )

        return { 'total': loss }
