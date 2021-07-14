import torch
import torch.nn as nn
import torch.nn.functional as F

from .init_net import init_net

class SpatialTransformationLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def build_net(self, x):
        n_channels = x.size(1)

        loc_conv = nn.Sequential(
            nn.Conv2d(n_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        loc_conv.to(x.device)
        test_out = loc_conv(x)

        loc_fc = nn.Sequential(
            nn.Linear( 10 * test_out.size(2) * test_out.size(3), 32),
            nn.ReLU(),
            nn.Linear( 32, 6 )
        )

        self.localization = nn.Sequential(
            loc_conv,
            nn.Flatten(start_dim=1),
            loc_fc
        )

        self.localization.to(x.device)

    def init_params(self, init_type, init_gain, gpu_ids):
        init_net(self, init_type, init_gain, gpu_ids)

        # Override initial parameters for final localization layer
        # Set them to an identity transformation
        self.localization[-1][-1].weight.data.zero_()
        self.localization[-1][-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.localization(x)
        theta = theta.view(-1,2,3)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)
