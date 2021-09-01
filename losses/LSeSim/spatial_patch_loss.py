import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import random

from Params import Params
from ..init_net import init_net
from .spatial_transformation_layer import SpatialTransformationLayer
from .patch_sim import PatchSim


class SpatialCorrelativeLoss(nn.Module):
    """
    learnable patch-based spatially-correlative loss with contrastive learning
    """
    def __init__(self, visualizer=None):
        super(SpatialCorrelativeLoss, self).__init__()
        self.patch_sim = PatchSim()
        self.use_attn = Params.loss['use_attn']
        self.attn_layer_types = Params.loss['attn_layer_types']
        self.attn_init_info = Params.loss['attn_init_info']
        self.loss_mode = Params.loss['ssim_compare_fn']
        self.T = Params.loss['T']
        self.criterion = nn.L1Loss() if Params.loss['use_norm'] else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.first_run = True
        self.visualizer = visualizer
        self.conv_layers = dict()


    def create_attn(self, feat, layer):
        """
        create the attention mapping layer used to transform features before building similarity maps
        :param feat: extracted features from a pretrained VGG or encoder for the similarity and dissimilarity map
        :return:
        """
        attn_layers = []
        conv_layers = []
        for layer_code in self.attn_layer_types:
            if layer_code == 'c':
                l = ConvAttentionLayer()
                attn_layers.append( l )
                conv_layers.append( l )
            elif layer_code == 's':
                attn_layers.append( SpatialTransformationLayer() )

        if not attn_layers:
            raise ValueError("attn_type must be a list of letters c or s" )

        for l in attn_layers:
            l.build_net(feat)
            feat = l(feat)
            l.init_params(self.init_type, self.init_gain, self.gpu_ids)

        # Extract the convolutional layers
        if conv_layers:
            self.conv_layers[layer] = nn.Sequential(*conv_layers)

        attn_net = nn.Sequential(*attn_layers)
        setattr(self, 'attn_%d' % layer, attn_net)


    def train_attn(self, real_A, fake_B, real_B):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
#        if self.opt.augment:
#            real_A = torch.cat([real_A, real_A], dim=0)
#            fake_B = torch.cat([fake_B.detach(), aug_A], dim=0)
#            real_B = torch.cat([real_B, aug_B], dim=0)
        for param in self.parameters():
            param.requires_grad = True
        self.optimizer.zero_grad()

        loss_spatial = self.loss(real_A, fake_B.detach(), real_B)
        loss_spatial.backward()
        self.optimizer.step()

        for param in self.parameters():
            param.requires_grad = False

    def cal_sim(self, f_src, f_tgt, f_other=None, patch_ids=None):
        """
        calculate the similarity map using the fixed/learned query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        output_sims = [None, None, None]

        for idx, feat in enumerate((f_src, f_tgt, f_other)):
            if feat is not None:
                feat = self.attn(feat) if self.use_attn else feat
                output_sims[idx], patch_ids = self.patch_sim(feat, patch_ids)

        return output_sims

    def compare_sim(self, sim_src, sim_tgt, sim_other):
        """
        measure the shape distance between the same shape and different inputs
        :param sim_src: the shape similarity map from source input image
        :param sim_tgt: the shape similarity map from target output image
        :param sim_other: the shape similarity map from other input image
        :return:
        """
        B, Num, N = sim_src.size()
        if self.loss_mode == 'info' or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            loss = self.cross_entropy_loss(sam_self, torch.arange(0, sam_self.size(0), dtype=torch.long, device=sim_src.device) % (Num))
        else:
            tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
            num = int(N / 4)
            src = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_src, sim_src)
            tgt = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_tgt, sim_tgt)
            if self.loss_mode == 'l1':
                loss = self.criterion((N / num) * src, (N / num) * tgt)
            elif self.loss_mode == 'cos':
                sim_pos = F.cosine_similarity(src, tgt, dim=-1)
                loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
            else:
                raise NotImplementedError('padding [%s] is not implemented' % self.loss_mode)

        return loss

    def loss(self, f_src, f_tgt, f_other=None):
        """
        calculate the spatial similarity and dissimilarity loss for given features from source and target domain
        :param f_src: source domain features
        :param f_tgt: target domain features
        :param f_other: other random sampled features
        :return:
        """
        sim_src, sim_tgt, sim_other = self.cal_sim(f_src, f_tgt, f_other)
        # calculate the spatial similarity for source and target domain
        loss = self.compare_sim(sim_src, sim_tgt, sim_other)
        if self.visualizer and self.visualizer.save_this_iter:
            chosen_patch_num = random.choice(range(Params.loss['num_patches']))

            for name, ssim in (('SSim_FakeB', sim_src), ('SSim_RealB', sim_tgt)):
                out_ssim = ssim[:,chosen_patch_num,:]
                out_ssim = out_ssim.view(-1, 1, Params.loss['patch_size'], Params.loss['patch_size'])
                self.visualizer.add_tensor( name, out_ssim )
        return loss

    def forward(self, realA, fakeB, realB, vgg_layer):
        if self.use_attn:
            if self.first_run:
                self.create_attn(fakeB, vgg_layer)
                self.optimizer = Params.create_optimizer(self.parameters())
                self.first_run = False
            self.train_attn(realA, fakeB, realB)

        return self.loss(fakeB, realB)


class ConvAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def build_net(self, feat):
        input_nc = feat.size(1)
        output_nc = max(32, input_nc // 4)
        self.net = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_nc, output_nc, kernel_size=1)
        )
        self.net.to(feat.device)

    def init_params(self, init_type, init_gain, gpu_ids):
        init_net(self, init_type, init_gain, gpu_ids)

    def forward(self, x):
        return self.net(x)
