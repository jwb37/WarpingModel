from pathlib import Path

from .FineGrainedSBIR import FineGrainedSBIR_Dataset


class ShoesDataset(FineGrainedSBIR_Dataset):
    def __init__( self, *args, **kwargs ):
        super().__init__(Path('datasets/ShoeV2/'), *args, **kwargs)
