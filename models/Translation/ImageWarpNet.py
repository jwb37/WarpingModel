import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

from ..BaseWarpModel import BaseWarpModel
from ..vgg19 import VGG19

from Params import Params
from losses import get_loss_module
from .WarpGenerator import WarpGenerator
from .Correlation import correlation_map


class ImageWarpNet(BaseWarpModel):
    def __init__(self, visualizer=None):
        super().__init__(VGG19, WarpGenerator, visualizer)

        self.calc_loss = get_loss_module()
        self.calc_loss.visualizer = visualizer
        self.trained_nets['calc_loss'] = self.calc_loss

        self.num_iterations = Params.NumWarpIterations

    #-------------------------------------------------------------------------
    # Internals

    def feats_to_flow(self, f1, f2):
        corr = correlation_map(f1, f2)
        return self.wg( corr )

    def warpTensor(self, tens, flow):
        B, _, H, W = flow.size()
        base_x = torch.linspace(-1,1,W).reshape(1, 1, W, 1).repeat(B,H,1,1)
        base_y = torch.linspace(-1,1,H).reshape(1, H, 1, 1).repeat(B,1,W,1)
        base_grid = torch.cat( (base_x, base_y), dim=-1 ).to(Params.Device)
        warp_grid = base_grid + flow.movedim(1,-1)

        return F.grid_sample(tens, warp_grid, align_corners=False)

    def calc_flow(self, tensorA, tensorB, return_feats=False):
        featsA = self.vggA(tensorA, ['pool4'] )
        featsB = self.vggB(tensorB, ['pool4'] )

        featsA = F.normalize(featsA['pool4']).detach()
        featsB = F.normalize(featsB['pool4']).detach()

        if return_feats:        
            original_feats = (featsA.clone(), featsB.clone())

        batch_size = tensorA.size(0)
        flow = self.feats_to_flow(featsA, featsB)

        for k in range(self.num_iterations):
            featsA = self.warpTensor(featsA, flow)
            featsA = F.normalize(featsA)
            flow = flow + self.feats_to_flow(featsA, featsB)

        if return_feats:
            return flow, original_feats[0], original_feats[1]
        else:
            return flow

    #---------------------------------------------------------------
    # Training functions

    def training_step(self, tensorA, tensorB, output_imgs=False):
        self.optimizer.zero_grad()

        flow, featsA, featsB = self.calc_flow(tensorA, tensorB, return_feats=True)
        warpedfeatsA = F.normalize(self.warpTensor(featsA, flow))
        loss = self.calc_loss(featsA, warpedfeatsA, featsB, flow)
        loss.backward()

#-- Code for warping whole image then extracting feats, rather than just warping extracted features
#        flow_interp = F.interpolate(flow, size=(256,256), mode='bilinear', align_corners=False)

#        warpedA = self.warpTensor(tensorA, flow_interp)
#        warpedfeatsA = self.vggA( warpedA, ['pool4'] )['pool4']
#        warpedfeatsA = F.normalize(warpedfeatsA)

#        losses = {
#            'patch':    self.calc_loss(featsA, warpedfeatsA, featsB, flow),
#            'flow_reg': torch.mean(torch.maximum( torch.abs(flow) - 1, torch.zeros_like(flow) ))
#        }
#        losses['final'] = losses['patch'] + losses['flow_reg']
#        losses['final'].backward()

        self.optimizer.step()

        if self.visualizer and self.visualizer.save_this_iter:
            self.visualizer.add_tensor( 'imgA', tensorA )
            self.visualizer.add_tensor( 'imgB', tensorB )
#            self.visualizer.add_tensor( 'warpedA', warpedA )
            self.visualizer.add_tensor( 'flow', flow )

            upscaled_flow = F.interpolate( flow, scale_factor=16 )
            warpedA = self.warpTensor( tensorA, upscaled_flow )
            self.visualizer.add_tensor( 'upscaled_flow', upscaled_flow )
            self.visualizer.add_tensor( 'warpedA', warpedA )
#            self.visualizer.add_tensor( 'flow_interp', flow_interp )

#        return losses['final'].item()
        return { 'total': loss }
