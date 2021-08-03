import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class VGG19(nn.Module):
    def __init__(self, savefile=None):
        super().__init__()

        net = models.vgg19(pretrained=(savefile is None))
        if savefile is not None:
            state_dict = torch.load( savefile )
            net.features.load_state_dict( state_dict )

        layer_names = [
            'conv1_1', 'relu1_1',
            'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1',
            'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2',
            'conv3_3', 'relu3_3',
            'conv3_4', 'relu3_4',
            'pool3',
            'conv4_1', 'relu4_1',
            'conv4_2', 'relu4_2',
            'conv4_3', 'relu4_3',
            'conv4_4', 'relu4_4',
            'pool4',
            'conv5_1', 'relu5_1'
        ]

        self.layers = { name: net.features[x] for x, name in enumerate(layer_names) }
        for name, layer in self.layers.items():
            setattr(self, name, layer)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, layers=None):
        out = {}
        for layer_name, func in self.layers.items():
            x = out[layer_name] = func(x)

        if len(layers) > 0:
            feats = {key: out[key] for key in layers}
            return feats
        else:
            return out['relu5_1']

if __name__ == '__main__':
    net = VGG19()
    test = torch.ones((32, 3, 256, 256))
    desired_layer = 'relu3_1'
    output = net(test, [desired_layer])
    print(output[desired_layer].shape)
    print( [(k, v.shape) for k, v in net.state_dict().items()] )
