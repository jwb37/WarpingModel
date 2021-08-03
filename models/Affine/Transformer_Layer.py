import torch
import torch.nn as nn
import itertools

from Params import Params

class Transformer_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, trs):
        [NumBatch,Z,H,W] = img.size()

        output = torch.zeros( (NumBatch, Z, H*3, W*3) ).to(Params.Device)

        sx = torch.linspace(-1,1,W).reshape(1, 1, W).repeat(NumBatch,H,1).to(Params.Device)
        sy = torch.linspace(-1,1,H).reshape(1, H, 1).repeat(NumBatch,1,W).to(Params.Device)

        for (i,j) in itertools.product(range(3),range(3)):
            xxyy_sx = trs[:,0,:,:] * (2/(H-1)) * (i-1) + trs[:,1,:,:] * (2/(W-1)) * (j-1) + trs[:,2,:,:] + sx
            xxyy_sy = trs[:,3,:,:] * (2/(H-1)) * (i-1) + trs[:,4,:,:] * (2/(W-1)) * (j-1) + trs[:,5,:,:] + sy
            # NumBatch x H x W x 2
            xxyy_s = torch.stack( (xxyy_sx, xxyy_sy), dim=3 )
            #xxyy_s = torch.stack( (xxyy_sy, xxyy_sx), dim=3 )

            output[:, :, i::3, j::3] = nn.functional.grid_sample( img, xxyy_s, align_corners=True )

        return output
