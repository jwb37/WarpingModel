import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

from Params import Params
from vgg19 import VGG19
from WarpGenerator import WarpGenerator
from Correlation import correlation_map
from losses import get_loss_module


class ImageWarpNet:
    def __init__(self, visualizer=None):
        self.vggA = VGG19()
        self.vggB = VGG19()
        self.wg = WarpGenerator()
        self.calc_loss = get_loss_module()

        self.visualizer = visualizer
        self.calc_loss.visualizer = visualizer

        for net in [self.vggA, self.vggB, self.wg]:
            net.to(Params.Device)

        # VGG models are pre-trained and only used for extracting features.
        for model in [self.vggA, self.vggB]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        self.num_iterations = Params.NumWarpIterations

        self.img_transform = transforms.Compose([
            transforms.Resize( (256, 256) ),
            transforms.Normalize( (123.680,116.779,103.939), (1., 1., 1.) ),
            transforms.ToTensor()
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
        flow = torch.zeros( (batch_size, 2, 16, 16) ).to(Params.Device)
        flow = flow + self.feats_to_flow(featsA, featsB)

        for k in range(self.num_iterations):
            featsA = self.warpTensor(featsA, flow)
            featsA = F.normalize(featsA)
            flow = flow + self.feats_to_flow(featsA, featsB)

        if return_feats:
            return flow, *original_feats
        else:
            return flow

    def warpImage(self, img, flow):
        orig_size = img.size

        img_transform = transforms.Compose([
            transforms.Resize( (256, 256) ),
            transforms.ToTensor()
        ])

        flow = F.interpolate(flow, size=(256,256), mode='bilinear')

        tens = img_transform(img).unsqueeze(0).to(Params.Device)
        tens = self.warpTensor(tens, flow)

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


    def training_step(self, tensorA, tensorB, output_imgs=False):
        self.optimizer.zero_grad()

        flow, featsA, featsB = self.calc_flow(tensorA, tensorB, return_feats=True)

        flow_interp = F.interpolate(flow, size=(256,256), mode='bilinear', align_corners=False)

        warpedA = self.warpTensor(tensorA, flow_interp)
        warpedfeatsA = self.vggA( warpedA, ['pool4'] )['pool4']
        warpedfeatsA = F.normalize(warpedfeatsA)

        losses = {
            'patch':    self.calc_loss(featsA, warpedfeatsA, featsB, flow),
            'flow_reg': torch.mean(torch.maximum( torch.abs(flow) - 1, torch.zeros_like(flow) ))
        }
        losses['final'] = losses['patch'] + losses['flow_reg']
        losses['final'].backward()
        self.optimizer.step()

        if self.visualizer:
            self.visualizer.add_tensor( 'tensorA', tensorA )
            self.visualizer.add_tensor( 'tensorB', tensorB )
            self.visualizer.add_tensor( 'featsA', featsA )
            self.visualizer.add_tensor( 'warpedA', warpedA )
            self.visualizer.add_tensor( 'warpedfeatsA', warpedfeatsA )
            self.visualizer.add_tensor( 'featsB', featsB )
            self.visualizer.add_tensor( 'flow', flow )
            self.visualizer.add_tensor( 'flow_interp', flow_interp )

        return losses['final'].item()
        
    #-----------------------------------------------------
    #  Save and load the model/optimizer parameters

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
