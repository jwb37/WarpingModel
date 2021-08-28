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
    def __init__(self, base_dir, phase, ret_tensor=True):
        self.all_data = []

        if 'A_suffixes' in Params.Dataset:
            self.A_suffixes = Params.Dataset['A_suffixes']
        else:
            self.A_suffixes = ['A']

        if 'B_suffix' in Params.Dataset:
            self.B_suffix = Params.Dataset['B_suffix']
        # Main time we need to override suffix is if training a Mimic network
        # Hardcode that case here so that we don't have to specify it in every experiment params file
        # (Note can always be overridden by a B_suffix setting in the params file)
        elif Params.ModelName == 'Mimic' and Params.isTrue('isTrain'):
            self.B_suffix = 'Flow'
        else:
            self.B_suffix = 'B'

        self.all_suffixes = self.A_suffixes + [self.B_suffix]

        dirs = dict()
        for suffix in self.all_suffixes:
            dirs[suffix] = base_dir / (phase + suffix)
            if 'overwrite_dir' in Params.Dataset:
                overwrite_dir = Path(Params.Dataset['overwrite_dir']) / (phase + suffix)
                if overwrite_dir.exists():
                    dirs[suffix] = overwrite_dir

        self.ret_tensor = ret_tensor

        self.re_strip_end = re.compile( r"(.*)_\d+(\.(?:png|jpg|jpeg))$" )

        self.all_data = [
            self.gen_data_point(dirs, sketch_path)
            for sketch_path in dirs[self.A_suffixes[0]].iterdir()
        ]

        self.transform = transforms.Compose( [
            transforms.Resize(Params.CropSize),
            transforms.RandomCrop( (Params.CropSize, Params.CropSize) ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ] )
        self.transform_greyscale = transforms.Compose( [
            transforms.Grayscale(1),
            transforms.Resize(Params.CropSize),
            transforms.RandomCrop( (Params.CropSize, Params.CropSize) ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ] )

        if not ret_tensor:
            self.A_transform = self.B_transform = torch.nn.Identity()
        else:
            self.A_transform = self.transform
            if hasattr(Params, 'InputNC') and Params.InputNC==len(self.A_suffixes):
                self.A_transform = self.transform_greyscale
            self.B_transform = self.transform


    def __len__(self):
        return len(self.all_data)


    def __getitem__(self, idx):
        data_pt = self.all_data[idx]

        raw_data = dict()
        for suffix in self.all_suffixes:
            fname = data_pt[suffix]
            if fname.suffix == '.pt':
                raw_data[suffix] = torch.load(fname).squeeze(dim=0)
            elif fname.suffix in ImageSuffices:
                img = Image.open( fname ).convert('RGB')
                raw_data[suffix] = img

        ans = dict()
        if len(self.A_suffixes) > 1:
            ans['imageA'] = torch.cat( [self.A_transform(raw_data[suffix]) for suffix in self.A_suffixes], dim=0 )
        else:
            ans['imageA'] = self.A_transform(raw_data[self.A_suffixes[0]])

        ans['imageB'] = self.B_transform(raw_data[self.B_suffix])
        ans['fname'] = data_pt['fname']

        return ans


    def strip_file_end(self, fname):
        m = self.re_strip_end.match(fname)
        if not m:
            return fname
        return ''.join(m.group(1,2))


    def gen_data_point(self, dirs, A_path):
        ''' This function exists to generate all other required paths to match a given 'A' path.
            This dataset class supports multiple A folders, so we first find the matching file in each of those.

            Also, for the B image, we need to find the correct filename in the 'B' folder:
            The structure of this dataset has multiple 'A' images for each 'B' image
            So that in the 'A' folder, files '001-1.png', '001-2.png' etc. match up
            with the single file '001.png' in the 'B' folder. We use a regular expression
            in the function 'strip_file_end' to achieve this.
        '''
        data_pt = { 'fname': A_path.name }
        data_pt.update({
            suffix: dirs[suffix] / A_path.name
            for suffix in self.A_suffixes
        })

        B_dir = dirs[self.B_suffix]
        B_path = B_dir / self.strip_file_end(A_path.name)

        # No such image file found... look for a saved '.pt' tensor instead
        if not B_path.exists():
            B_path = B_dir / (A_path.stem + '.pt')

        data_pt[self.B_suffix] = B_path

        return data_pt
