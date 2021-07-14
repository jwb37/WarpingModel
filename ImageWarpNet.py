import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

import Params
from vgg19 import VGG19
from WarpGenerator import WarpGenerator
from Correlation import correlation_map


class ImageWarpNet:
    def __init__(self, num_iterations=3):
        self.vggA = VGG19()
        self.vggB = VGG19()
        self.wg = WarpGenerator()

        for net in [self.vggA, self.vggB, self.wg]:
            net.to(Params.Device)

        # VGG models are pre-trained and only used for extracting features.
        for model in [self.vggA, self.vggB]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        self.num_iterations = num_iterations

        self.img_transform = transforms.Compose([
            transforms.Resize( (256, 256) ),
            transforms.ToTensor(),
            transforms.Normalize( (123.680,116.779,103.939), (1., 1., 1.) )
        ])

    #------------------------------------------------------------------------
    # Main public method(s) for trained model
    def warp(self, imgA, imgB):
        """ Warps image A to match the geometric structure of image B"""

        tensA = self.img_transform(imgA).unsqueeze(0).to(Params.Device)
        tensB = self.img_transform(imgB).unsqueeze(0).to(Params.Device)

        flow = self.calc_flow(tensA, tensB)
        flow = flow.movedim(1, -1)

        return self.warpImage(imgA, flow)


    #-------------------------------------------------------------------------
    # Internals

    def feats_to_warp_grid(self, f1, f2):
        corr = correlation_map(f1, f2)
        return self.wg( corr )

    def warpTensor(self, tens, warp_grid):
        return F.grid_sample(tens, warp_grid.movedim(1,-1), align_corners=False)

    def calc_flow(self, tensorA, tensorB, return_feats=False):
        featsA = self.vggA(tensorA, ['pool4'] )
        featsB = self.vggB(tensorB, ['pool4'] )

        featsA = F.normalize(featsA['pool4']).detach()
        featsB = F.normalize(featsB['pool4']).detach()

        if return_feats:        
            original_feats = (featsA.clone(), featsB.clone())

        batch_size = tensorA.size(0)
        warp_grid = torch.zeros( (batch_size, 2, 16, 16) ).to(Params.Device)
        warp_grid = warp_grid + self.feats_to_warp_grid(featsA, featsB)

        for k in range(self.num_iterations):
            featsA = self.warpTensor(featsA, warp_grid)
            warp_grid = warp_grid + self.feats_to_warp_grid(featsA, featsB)

        if return_feats:
            return warp_grid, *original_feats
        else:
            return warp_grid

    def warpImage(self, img, flow):
        orig_size = img.size

        img_transform = transforms.Compose([
            transforms.Resize( (256, 256) ),
            transforms.ToTensor()
        ])

        flow = F.interpolate(flow, size=(256,256), mode='bilinear')

        tens = img_transform(img).unsqueeze(0).to(Params.Device)
        tens = F.grid_sample(tens, flow, align_corners=False)

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
        self.optimizer = Params.create_optimizer(self.wg.parameters())
        self.wg.train()


    def calc_loss(self, tensorA, tensorB, warp_grid, patch_size=9):
        warpedA = self.warpTensor(tensorA, warp_grid)
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


    def training_step(self, tensorA, tensorB):
        self.optimizer.zero_grad()

        warp_grid, featsA, featsB = self.calc_flow(tensorA, tensorB, return_feats=True)

        loss = self.calc_loss(featsA, featsB, warp_grid)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    #-----------------------------------------------------
    def save(self, filename):
        save_state = {
            'model': self.wg.state_dict(),
            'optim': self.optimizer.state_dict()
        }
        torch.save( save_state, filename )

    def load(self, filename):
        save_state = torch.load( filename )
        self.wg.load_state_dict( save_state['model'] )
        self.optimizer.load_state_dict( save_state['optim'] )
