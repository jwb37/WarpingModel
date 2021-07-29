from Params import Params
from .LSeSim.spatial_patch_loss import SpatialCorrelativeLoss
from .WarpStyleTransfer.WST_Loss import WST_Loss

def get_loss_module():
    requested_type = Params.loss['type'].lower()
    if requested_type == 'lsesim':
        return SpatialCorrelativeLoss()
    elif requested_type == 'wst':
        return WST_Loss()
