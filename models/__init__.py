from Params import Params
from .Affine.ImageWarpNet import ImageWarpNet as AffineModel
from .Translation.ImageWarpNet import ImageWarpNet as TranslationModel

def get_model(visualizer=None):
    if Params.ModelName == 'Affine':
        return AffineModel(visualizer)
    elif Params.ModelName == 'Translation':
        return TranslationModel(visualizer)
