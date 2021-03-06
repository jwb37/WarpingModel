import torch
import torch.nn as nn

class WarpGenerator(nn.Module):
    def __init__(self):
        super(WarpGenerator,self).__init__()
        self.create_layers()
    
    def create_layers(self):
        self.geo_conv1 = nn.Conv2d(81, 64, (9,9), stride=1, padding=4, dilation=1)
        self.geo_relu1 = nn.ReLU()
        self.geo_conv1b = nn.Conv2d(64, 64, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu1b = nn.ReLU()
        self.geo_pooling1 = nn.Sequential( nn.ConstantPad2d((0, 1, 0, 1), 0), nn.MaxPool2d( (2, 2), stride=(2, 2)) )
        self.geo_conv2 = nn.Conv2d(64, 128, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu2 = nn.ReLU()
        self.geo_conv2b = nn.Conv2d(128, 128, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu2b = nn.ReLU()
        self.geo_pooling2 = nn.Sequential( nn.ConstantPad2d((0, 1, 0, 1), 0), nn.MaxPool2d( (2, 2), stride=(2, 2)) )
        self.geo_conv3 = nn.Conv2d(128, 256, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu3 = nn.ReLU()
        self.geo_conv3b = nn.Conv2d(256, 256, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu3b = nn.ReLU()
        self.geo_pooling3 = nn.Sequential( nn.ConstantPad2d((0, 1, 0, 1), 0), nn.MaxPool2d( (2, 2), stride=(2, 2)) )
        self.geo_conv4 = nn.Conv2d(256, 512, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu4 = nn.ReLU()
        self.geo_conv4b = nn.Conv2d(512, 512, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu4b = nn.ReLU()
        self.geo_pooling4 = nn.Sequential( nn.ConstantPad2d((0, 1, 0, 1), 0), nn.MaxPool2d( (2, 2), stride=(2, 2)) )
        self.geo_conv5 = nn.Conv2d(512, 512, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu5 = nn.ReLU()
        self.geo_conv5b = nn.Conv2d(512, 512, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu5b = nn.ReLU()
        self.geo_deconv1 = nn.ConvTranspose2d(512, 512, (4, 4), stride=(2, 2), bias=False)
        self.geo_conv6 = nn.Conv2d(1024, 512, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu6 = nn.ReLU()
        self.geo_conv6b = nn.Conv2d(512, 512, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu6b = nn.ReLU()
        self.geo_deconv2 = nn.ConvTranspose2d(512, 512, (4, 4), stride=(2, 2), bias=False)
        self.geo_conv7 = nn.Conv2d(768, 256, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu7 = nn.ReLU()
        self.geo_conv7b = nn.Conv2d(256, 256, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu7b = nn.ReLU()
        self.geo_deconv3 = nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2), bias=False)
        self.geo_conv8 = nn.Conv2d(384, 128, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu8 = nn.ReLU()
        self.geo_conv8b = nn.Conv2d(128, 128, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu8b = nn.ReLU()
        self.geo_deconv4 = nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2), bias=False)
        self.geo_conv9 = nn.Conv2d(192, 64, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu9 = nn.ReLU()
        self.geo_conv9b = nn.Conv2d(64, 64, (3,3), stride=1, padding=1, dilation=1)
        self.geo_relu9b = nn.ReLU()
        self.geo_conv10 = nn.Conv2d(64, 6, (9,9), stride=1, padding=4, dilation=1)
    
    def forward(self, init_corr_vol_norm):
        init_x1 = self.geo_conv1(init_corr_vol_norm)
        init_x2 = self.geo_relu1(init_x1)
        init_x3 = self.geo_conv1b(init_x2)
        init_x4 = self.geo_relu1b(init_x3)
        init_x5 = self.geo_pooling1(init_x4)
        init_x6 = self.geo_conv2(init_x5)
        init_x7 = self.geo_relu2(init_x6)
        init_x8 = self.geo_conv2b(init_x7)
        init_x9 = self.geo_relu2b(init_x8)
        init_x10 = self.geo_pooling2(init_x9)
        init_x11 = self.geo_conv3(init_x10)
        init_x12 = self.geo_relu3(init_x11)
        init_x13 = self.geo_conv3b(init_x12)
        init_x14 = self.geo_relu3b(init_x13)
        init_x15 = self.geo_pooling3(init_x14)
        init_x16 = self.geo_conv4(init_x15)
        init_x17 = self.geo_relu4(init_x16)
        init_x18 = self.geo_conv4b(init_x17)
        init_x19 = self.geo_relu4b(init_x18)
        init_x20 = self.geo_pooling4(init_x19)
        init_x21 = self.geo_conv5(init_x20)
        init_x22 = self.geo_relu5(init_x21)
        init_x23 = self.geo_conv5b(init_x22)
        init_x24 = self.geo_relu5b(init_x23)
        init_x25 = self.geo_deconv1(init_x24)
        init_x25 = init_x25[:,:,1:-1,1:-1]
        init_x26 = torch.cat( (init_x25,init_x19), dim=1 )
        init_x27 = self.geo_conv6(init_x26)
        init_x28 = self.geo_relu6(init_x27)
        init_x29 = self.geo_conv6b(init_x28)
        init_x30 = self.geo_relu6b(init_x29)
        init_x31 = self.geo_deconv2(init_x30)
        init_x31 = init_x31[:,:,1:-1,1:-1]
        init_x32 = torch.cat( (init_x31,init_x14), dim=1 )
        init_x33 = self.geo_conv7(init_x32)
        init_x34 = self.geo_relu7(init_x33)
        init_x35 = self.geo_conv7b(init_x34)
        init_x36 = self.geo_relu7b(init_x35)
        init_x37 = self.geo_deconv3(init_x36)
        init_x37 = init_x37[:,:,1:-1,1:-1]
        init_x38 = torch.cat( (init_x37,init_x9), dim=1 )
        init_x39 = self.geo_conv8(init_x38)
        init_x40 = self.geo_relu8(init_x39)
        init_x41 = self.geo_conv8b(init_x40)
        init_x42 = self.geo_relu8b(init_x41)
        init_x43 = self.geo_deconv4(init_x42)
        init_x43 = init_x43[:,:,1:-1,1:-1]
        init_x44 = torch.cat( (init_x43,init_x4), dim=1 )
        init_x45 = self.geo_conv9(init_x44)
        init_x46 = self.geo_relu9(init_x45)
        init_x47 = self.geo_conv9b(init_x46)
        init_x48 = self.geo_relu9b(init_x47)
        init_aff_transform = self.geo_conv10(init_x48)
        return init_aff_transform
