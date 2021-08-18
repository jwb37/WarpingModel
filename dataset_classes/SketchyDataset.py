import re
from collections import namedtuple
from pathlib import Path


from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


from Params import Params

DataPoint = namedtuple('DataPoint', ('sketch', 'photo', 'category'))

class SketchyDataset(Dataset):
    def __init__(self, phase, ret_tensor=True):
        self.all_data = []

        self.ret_tensor = ret_tensor

        base_path = Path( './datasets/Sketchy/', phase )

        sketch_subdirs = [entry for entry in (base_path/'sketch').iterdir() if entry.is_dir()]
        sketch_subdirs.sort(key=lambda e: e.name)

        sketch_re = re.compile( r"^(.*)-\d+\.png$" )

        for sketch_subdir in sketch_subdirs:
            photo_subdir = base_path/'photo'/sketch_subdir.name
            for sketch_path in sketch_subdir.iterdir():
                m = sketch_re.match(sketch_path.name)
                if m is not None:
                    self.all_data.append(
                        DataPoint(
                            sketch = sketch_path,
                            photo = photo_subdir/f"{m.group(1)}.jpg",
                            category = sketch_subdir.name
                        )
                    )

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

        ans = { 'category': data_pt.category, 'fname': data_pt.sketch.name }
        for out_name, fname in zip( ('imageA', 'imageB'), (data_pt.sketch, data_pt.photo) ):
            img = Image.open( fname ).convert('RGB')
            if self.ret_tensor:
                ans[out_name] = self.transform(img)
            else:
                ans[out_name] = img

        return ans
