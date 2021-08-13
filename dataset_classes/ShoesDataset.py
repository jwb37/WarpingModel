from .FineGrainedSBIR import FineGrainedSBIR_Dataset

class ShoesDataset(FineGrainedSBIR_Dataset):
    def __init__( *args, **kwargs ):
        super().__init__('datasets/Shoes/', *args, **kwargs)
