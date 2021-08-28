import torch
import torch.nn.functional as F

from .pix2pix_model import Pix2PixModel

class Pix2PixFlowModel(Pix2PixModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.visual_names += ['warpedA']

    def warpTensor(self, tens, flow):
        B, _, H, W = flow.size()
        base_x = torch.linspace(-1,1,W).reshape(1, 1, W, 1).repeat(B,H,1,1)
        base_y = torch.linspace(-1,1,H).reshape(1, H, 1, 1).repeat(B,1,W,1)
        base_grid = torch.cat( (base_x, base_y), dim=-1 ).to(self.device)
        warp_grid = base_grid + flow.movedim(1,-1)

        return F.grid_sample(tens, warp_grid, align_corners=False)

    def __call__(self, realA):
        super().__call__(realA)
        return self.warpTensor(self.real_A, self.fake_B)
