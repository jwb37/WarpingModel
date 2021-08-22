import re
import torch
from collections import namedtuple
from pathlib import Path


from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


from Params import Params


DataPoint = namedtuple('DataPoint', ('sketch', 'photo', 'fname'))

ImageSuffices = ( '.png', '.jpg', '.jpeg', '.bmp', '.tiff' )


class FineGrainedSBIR_Dataset(Dataset):
    def __init__(self, base_dir, phase, A_suffix='A', B_suffix='B', ret_tensor=True):
        self.all_data = []

        if 'B_suffix' in Params.Dataset:
            B_suffix = Params.Dataset['B_suffix']

        dirs = dict()
        for key, suffix in zip(('sketch','photo'), (A_suffix, B_suffix)):
            dirs[key] = base_dir / (phase + suffix)
            if 'overwrite_dir' in Params.Dataset:
                overwrite_dir = Path(Params.Dataset['overwrite_dir']) / (phase + suffix)
                if overwrite_dir.exists():
                    dirs[key] = overwrite_dir

        self.ret_tensor = ret_tensor

        self.re_strip_end = re.compile( r"(.*)_\d+(\.(?:png|jpg|jpeg))$" )

        self.all_data = [
            self.gen_data_point(dirs['photo'], sketch_path)
            for sketch_path in dirs['sketch'].iterdir()
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
            elif fname.suffix in ImageSuffices:
                img = Image.open( fname ).convert('RGB')
                if self.ret_tensor:
                    img = self.transform(img)
                ans[out_name] = img

        return ans


    def strip_file_end(self, fname):
        m = self.re_strip_end.match(fname)
        if not m:
            return fname
        return ''.join(m.group(1,2))


    def gen_data_point(self, photo_dir, sketch_path):
        photo_path = photo_dir / self.strip_file_end(sketch_path.name)

        # No such image file found... look for a saved '.pt' tensor instead
        if not photo_path.exists():
            photo_path = photo_dir / (sketch_path.stem + '.pt')

        return DataPoint( sketch_path, photo_path, sketch_path.name )
