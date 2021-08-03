import torch
import torch.nn as nn
import torch.nn.functional as F

class DeconvBlock(nn.Module):
    def __init__(self):
        super(DeconvBlock,self).__init__()
#        self.iter1_deconv = nn.ConvTranspose2d( 6, 6, (4,4), stride=(2,2), bias=False )
#        self.final_deconv = nn.ConvTranspose2d( 2, 2, (4,4), stride=(2,2), bias=False )

    def iter1(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #return self.iter1_deconv(x)[:,:,1:-1,1:-1]

    def final(self, x):
        return F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
#        x = self.final_deconv(x)[:,:,1:-1,1:-1]
#        x = self.final_deconv(x)[:,:,1:-1,1:-1]
#        return self.final_deconv(x)[:,:,1:-1,1:-1]

if __name__ == '__main__':
    net = DeconvBlock()
    print( [(k, v.shape) for k, v in net.state_dict().items()] )
