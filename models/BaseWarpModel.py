import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

from Params import Params
from losses import get_loss_module

from .BaseModel import BaseModel


class BaseWarpModel(BaseModel):

    def __init__(self, VGG_constr, WG_constr, visualizer=None):
        super().__init__(visualizer=visualizer)

        self.vggA = VGG_constr()
        self.vggB = VGG_constr()
        self.wg = WG_constr()

        for net in [self.vggA, self.vggB, self.wg]:
            net.to(Params.Device)

        self.trained_nets['model'] = self.wg

        if Params.isTrue('TrainVGG'):
            # Add VGG networks to set of networks to be saved/loaded
            self.trained_nets.update({
                'vggA': self.vggA,
                'vggB': self.vggB
            })
            # Set optimizer parameters to optimize over all 3 networks
            param_list = [net.parameters() for net in (self.wg, self.vggA, self.vggB)]
            optim_params = itertools.chain(*param_list)
        else:
            # VGG models are pre-trained and only used for extracting features.
            for model in [self.vggA, self.vggB]:
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
            # Set optimizer parameters to ONLY the warp generator
            optim_params = self.wg.parameters()

        self.optimizer = Params.create_optimizer(optim_params)
        self.optimizers['optim'] = self.optimizer

        self.img_transform = transforms.Compose([
            transforms.Resize( (256, 256) ),
            transforms.ToTensor(),
            transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
        ])

    #------------------------------------------------------------------------
    # Main public method(s) for trained model
    def warp(self, imgA, imgB):
        """ Warps image A to match the geometric structure of image B"""

        tensA = self.img_transform(imgA).unsqueeze(0).to(Params.Device)
        tensB = self.img_transform(imgB).unsqueeze(0).to(Params.Device)

        flow = self.calc_flow(tensA, tensB)

        return self.warpImage(imgA, flow)


    #-------------------------------------------------------------------------
    # Internals

    def calc_flow(self, tensorA, tensorB, return_feats=False):
        pass

    def warpImage(self, img, flow):
        orig_size = img.size

        img_transform = transforms.Compose([
            transforms.Resize( (256, 256) ),
            transforms.ToTensor()
        ])

        if Params.isTrue('BlurFlow'):
            flow = transforms.functional.gaussian_blur(flow, kernel_size=7)

        flow = F.interpolate(flow, size=(256,256), mode='bilinear')
        B, _, H, W = flow.size()
        base_x = torch.linspace(-1,1,W).reshape(1, 1, W, 1).repeat(B,H,1,1)
        base_y = torch.linspace(-1,1,H).reshape(1, H, 1, 1).repeat(B,1,W,1)
        base_grid = torch.cat( (base_x, base_y), dim=-1 ).to(Params.Device)
        warp_grid = base_grid + flow.movedim(1,-1)

        tens = img_transform(img).unsqueeze(0).to(Params.Device)
        tens = F.grid_sample(tens, warp_grid, align_corners=False)

        tens = tens[0,:,:,:]
        # PIL likes to have dimensions in order H x W x C
        tens = torch.movedim(tens,0,-1)

        out_img = tens.detach().cpu().numpy() * 255.0
        out_img = out_img.astype(np.uint8)
        out_img = Image.fromarray(out_img, mode="RGB")

        return out_img.resize( orig_size )


    #---------------------------------------------------------------
    # Training functions

    def prepare_training(self):
        if hasattr(self, 'calc_loss'):
            self.calc_loss.train()

    def prepare_testing(self):
        pass

    def training_step(self, tensorA, tensorB):
        pass
