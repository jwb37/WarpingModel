import torch
import torch.nn as nn
import itertools


from Params import Params


class Constraint_Correlation_Layer(nn.Module):
    def __init__(self, max_disp=9, stride=1):
        super().__init__()
        self.max_disp = max_disp
        self.stride = stride


    def forward(self, imgA, imgB):
        [NumBatch,Z,H,W] = imgA.size()
        center_max_disp = ((self.max_disp+1)/2) - 1

        output = torch.zeros( (NumBatch, self.max_disp**2, H, W) ).to(Params.Device)

        sx = torch.linspace(-1,1,W).reshape(1, 1, W).repeat(NumBatch, H, 1).to(Params.Device)
        sy = torch.linspace(-1,1,H).reshape(1, H, 1).repeat(NumBatch, 1, W).to(Params.Device)

        for winsize, (i,j) in enumerate(itertools.product(range(self.max_disp), range(self.max_disp))):
            xxyy_sx = sx + (2/(H-1))*(i-center_max_disp)*self.stride
            xxyy_sy = sy + (2/(W-1))*(j-center_max_disp)*self.stride

            # NumBatch x H x W x 2
            xxyy_s = torch.stack( (xxyy_sx, xxyy_sy), dim=3 ).to(Params.Device)
            #xxyy_s = torch.stack( (xxyy_sy, xxyy_sx), dim=3 )

            inputs2 = nn.functional.grid_sample( imgB, xxyy_s, align_corners=True )
            output[:,winsize,:,:] = torch.sum(imgA * inputs2, dim=1)

        return output
