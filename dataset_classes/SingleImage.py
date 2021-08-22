import re
import torch
from collections import namedtuple
from pathlib import Path


from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


from Params import Params

class SingleImageDataset(Dataset):
    def __init__(self, phase, ret_tensor=True):
        super().__init__()

#        if phase != 'test':
#            raise Exception("Error: Singe Image Dataset is only intended for testing, not for training")

        base_dir = Path('datasets', 'Single')

        self.transform = transforms.Compose( [
            transforms.Resize(Params.CropSize),
            transforms.RandomCrop( (Params.CropSize, Params.CropSize) ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ] )

        self.imageA = Image.open(base_dir / 'A.png').convert('RGB')
        self.imageB = Image.open(base_dir / 'B.png').convert('RGB')

        if ret_tensor:
            self.imageA = self.transform(self.imageA)
            self.imageB = self.transform(self.imageB)

        self.datapoint = {
            'imageA': self.imageA,
            'imageB': self.imageB,
            'fname': 'output.png'
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.datapoint
