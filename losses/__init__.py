from .LSeSim.spatial_patch_loss import SpatialCorrelativeLoss
from .WarpStyleTransfer.WST_Loss import WST_Loss

def get_loss_module():
    # Just a stub for now
    return SpatialCorrelativeLoss()
