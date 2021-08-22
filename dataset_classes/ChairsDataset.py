from pathlib import Path

from .FineGrainedSBIR import FineGrainedSBIR_Dataset


class ChairsDataset(FineGrainedSBIR_Dataset):
    def __init__( self, *args, **kwargs ):
        super().__init__(Path('datasets/ChairV2/'), *args, **kwargs)
