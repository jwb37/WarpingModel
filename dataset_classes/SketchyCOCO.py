import re
import torch
from collections import namedtuple
from pathlib import Path


from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


from Params import Params


DataPoint = namedtuple('DataPoint', ('sketch', 'photo', 'fname'))


def make_square(img):
    (w, h) = img.size
    padding = (abs(w - h) // 2, 0)
    if h > w:
        padding = tuple(reversed(padding))

    return transforms.functional.pad(img, padding, fill=255)

class SketchyCOCO_Dataset(Dataset):
    def __init__(self, phase, A_subfolder='Sketch', B_subfolder='GT', ret_tensor=True):
        self.all_data = []

        base_dir = Path('./datasets/SketchyCOCO/Object/')

        if hasattr(Params, 'B_subfolder'):
            B_subfolder = Params.B_subfolder

        # SketchyCOCO uses a test/validation split rather than a test/train split
        if phase == 'test':
            phase = 'val'

        sketch_dir = base_dir / A_subfolder / phase
        photo_dir = base_dir / B_subfolder / phase

        self.ret_tensor = ret_tensor

        for category in Params.SketchyCOCO_Categories:
            sketch_cat_dir = sketch_dir / str(category)
            photo_cat_dir = photo_dir / str(category)

            self.all_data += [
                DataPoint(
                    sketch = sketch_path,
                    photo = photo_cat_dir/sketch_path.name if (photo_cat_dir/sketch_path.name).exists() else photo_cat_dir/(sketch_path.stem + '.pt'),
                    fname = sketch_path.name
                )
                for sketch_path in sketch_cat_dir.iterdir()
                if not sketch_path.name.startswith('.')
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
            if fname.suffix == '.pt':
                ans[out_name] = torch.load(fname).squeeze(dim=0)
            elif fname.suffix == '.png':
                img = Image.open( fname ).convert('RGB')
                img = make_square(img)
                if self.ret_tensor:
                    img = self.transform(img)
                ans[out_name] = img

        return ans
