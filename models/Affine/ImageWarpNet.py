import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

from Params import Params

from ..vgg19 import VGG19
from ..BaseWarpModel import BaseWarpModel

from .DeconvBlock import DeconvBlock
from .Flow_Layer import Flow_Layer
from .L2Norm_Layer import L2Norm_Layer
from .WarpGenerator import WarpGenerator
from .Transformer_Layer import Transformer_Layer
from .Constraint_Correlation_Layer import Constraint_Correlation_Layer

from losses.RTN import RTN_Loss
from losses.Smoothness import SmoothnessLoss


class ImageWarpNet(BaseWarpModel):
    def __init__(self, visualizer=None):
        super().__init__(VGG19, WarpGenerator, visualizer)

        self.norm = F.normalize
        self.correlation = Constraint_Correlation_Layer()
        self.transformer = Transformer_Layer()
        self.flow_layer = Flow_Layer()
        self.deconv_block = DeconvBlock()
        self.deconv_block.to(Params.Device)

        self.RTN_loss = RTN_Loss()
        self.smoothness_loss = SmoothnessLoss()

        self.num_iterations = Params.NumWarpIterations


    def transform_to_flow(self, x):
        return torch.stack( (x[:,2,:,:], x[:,5,:,:]), dim=1 )


    def feats_to_aff_transform(self, f1, f2, return_corr=False):
        corr_vol = self.correlation(f1, f2)
        corr_vol_norm = self.norm(corr_vol)
        aff_transform = self.wg( corr_vol_norm )
        if return_corr:
            return aff_transform, corr_vol
        else:
            return aff_transform


    def calc_flow(self, tensorA, tensorB, return_final_corr=False):
        # Remaining code is adapted from RTN code, which warps tensor B onto tensor A
        # We opt to use the opposite convention.
        tensorA, tensorB = tensorB, tensorA

        featsA = self.vggA(tensorA, ['relu4_1', 'relu5_1'] )
        featsB = self.vggB(tensorB, ['relu5_1', 'pool3', 'pool4'] )

        f1_x21_norm = self.norm(featsA['relu4_1'])
        f1_x30_norm = self.norm(featsA['relu5_1'])
        f2_x30_norm = self.norm(featsB['relu5_1'])

        # Initialize warp field
        comp_aff_transform = torch.zeros( (1, 6, 16, 16) ).to(Params.Device)
        comp_aff_transform[0,0,:,:] = 1
        comp_aff_transform[0,4,:,:] = 1

        comp_aff_transform = comp_aff_transform + self.feats_to_aff_transform(f1_x30_norm, f2_x30_norm)

        f2 = self.transformer(featsB['pool4'], comp_aff_transform)
        f2 = self.vggB.layers['conv5_1'](f2)
        f2 = self.vggB.layers['relu5_1'](f2)
        f2 = self.norm(f2)

        comp_aff_transform = comp_aff_transform + self.feats_to_aff_transform(f1_x30_norm, f2)
        comp_aff_transform = self.deconv_block.iter1(comp_aff_transform)

        for k in range(self.num_iterations):
            f2 = self.transformer(featsB['pool3'], comp_aff_transform)
            f2 = self.vggB.layers['conv4_1'](f2)
            f2 = self.vggB.layers['relu4_1'](f2)
            f2 = self.norm(f2)

            aff_transform_diff, corr_vol = self.feats_to_aff_transform(f1_x21_norm, f2, True)
            comp_aff_transform = comp_aff_transform + aff_transform_diff

        flow = self.transform_to_flow(comp_aff_transform)

        # Query - geo_warping.m switches channels 1 and 2 around (x and y components of the flow)?
        #flow = torch.flip(flow, dims=(-1,))
        if return_final_corr:
            return flow, corr_vol
        else:
            return flow

    def training_step(self, tensorA, tensorB):
        self.optimizer.zero_grad()

        flow, corr_vol = self.calc_flow(tensorA, tensorB, True)

        losses = {
            'corr': self.RTN_loss(corr_vol)
        }

        if Params.isTrue('UseSmoothLoss'):
            losses['smooth'] = self.smoothness_loss(flow) * Params.SmoothLossLambda

        losses['total'] = sum(losses.values())
        losses['total'].backward()
        self.optimizer.step()

        if self.visualizer.save_this_iter:
            flow = self.deconv_block.final(flow)
            warpedA = self.flow_layer(tensorA, flow)
            self.visualizer.add_tensor('warpedA', warpedA)
            self.visualizer.add_tensor('imgA', tensorA)
            self.visualizer.add_tensor('imgB', tensorB)
            self.visualizer.add_tensor('flow', flow)

        return losses
