import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import Params
from ..init_net import init_net


class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]       # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = patch_ids // W, patch_ids % W
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    feat_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W*H)
        else:
            feat_query = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids


class SpatialCorrelativeLoss(nn.Module):
    """
    learnable patch-based spatially-correlative loss with contrastive learning
    """
    def __init__(self, loss_mode='cos', patch_nums=64, patch_size=9, norm=True, use_attn=True,
                 init_type='normal', init_gain=0.02, gpu_ids=[], T=0.1):
        super(SpatialCorrelativeLoss, self).__init__()
        self.patch_sim = PatchSim(patch_nums=patch_nums, patch_size=patch_size, norm=norm)
        self.patch_size = patch_size
        self.patch_nums = patch_nums
        self.norm = norm
        self.use_attn = use_attn
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.loss_mode = loss_mode
        self.T = T
        self.criterion = nn.L1Loss() if norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.first_run = True

    def create_attn(self, feat):
        """
        create the attention mapping layer used to transform features before building similarity maps
        :param feat: extracted features from a pretrained VGG or encoder for the similarity and dissimilarity map
        :return:
        """
        net = ConvAttentionLayer()
        net.build_net(feat)
        net.init_params(self.init_type, self.init_gain, self.gpu_ids)
        self.attn = net

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
        return loss

    def forward(self, realA, warpedA, realB, warp_grid):
        if self.first_run:
            self.create_attn(warpedA)
            self.optimizer = Params.create_optimizer(self.parameters())
            self.first_run = False
        self.train_attn(realA, warpedA, realB)
        return self.loss(warpedA, realB)


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
