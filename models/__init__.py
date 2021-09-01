from Params import Params
from .Mimic.ImageWarpNet import ImageWarpNet as MimicModel
from .Affine.ImageWarpNet import ImageWarpNet as AffineModel
from .Pix2Pix.pix2pix_model import Pix2PixModel

def get_model(visualizer=None):
    if Params.ModelName == 'Affine':
        return AffineModel(visualizer)
    elif Params.ModelName == 'Mimic':
        return MimicModel(visualizer)
    elif Params.ModelName == 'Pix2Pix':
        return Pix2PixModel(visualizer)
