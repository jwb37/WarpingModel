import torch
import torch.nn as nn

class Upsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose2d( 2, 2, (4,4), stride=(2,2), bias=False )

    def forward(self, x):
        for i in range(4):
            x = self.deconv(x)[:,:,1:-1,1:-1]
        return x

if __name__ == '__main__':
    net = Upsampler()
    print( [(k, v.shape) for k, v in net.state_dict().items()] )
