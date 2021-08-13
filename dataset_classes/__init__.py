from .ChairsDataset import ChairsDataset
from .ShoesDataset import ShoesDataset
from .SketchyDataset import SketchyDataset
from .SketchyCOCO import SketchyCOCO_Dataset

from Params import Params


create_dataset = {
    'Sketchy': SketchyDataset,
    'Shoes': ShoesDataset,
    'SketchyCOCO': SketchyCOCO_Dataset,
    'Chairs': ChairsDataset
}[Params.Dataset]
