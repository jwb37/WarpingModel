import re
from collections import namedtuple
from pathlib import Path


from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


from Params import Params


DataPoint = namedtuple('DataPoint', ('sketch', 'photo', 'fname'))


class ShoesDataset(Dataset):
    def __init__(self, phase):
        self.all_data = []

        base_dir = Path('./Shoes')

        if phase == 'train':
            sketch_dir = base_dir / 'trainA'
            photo_dir = base_dir / 'trainB'
        elif phase == 'test':
            sketch_dir = base_dir / 'testA'
            photo_dir = base_dir / 'testB'

        self.phase = phase

        self.all_data = [
            DataPoint(
                sketch = str(sketch_path),
                photo = str(photo_dir/sketch_path.name),
                fname = sketch_path.name
            )
            for sketch_path in sketch_dir.iterdir()
        ]

        self.transform = transforms.Compose( [
            transforms.Resize(Params.CropSize),
            transforms.RandomCrop( (Params.CropSize, Params.CropSize) ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ] )

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data_pt = self.all_data[idx]

        ans = {'fname': data_pt.fname}
        for out_name, fname in zip( ('imageA', 'imageB'), (data_pt.sketch, data_pt.photo) ):
            img = Image.open( fname ).convert('RGB')
            if self.phase == 'test':
                ans[out_name] = img
            elif self.phase == 'train':
                ans[out_name] = self.transform(img)

        return ans
