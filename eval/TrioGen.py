import torch
from PIL import Image
from tqdm import tqdm

from .BaseEval import BaseEval
from dataset_classes import create_dataset

import os
import os.path as path

from Params import Params


class TrioGen(BaseEval):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        self.dataset = {
            'train': create_dataset('train', ret_tensor=False),
            'test': create_dataset('test', ret_tensor=False)
        }

    def run_test(self):
        print("Testing")

        out_size = (256, 256)
        out_img = Image.new('RGB', (out_size[0] * 3, out_size[1]))

        with torch.no_grad():
            self.model.prepare_testing()
            for setname in ('train', 'test'):
                out_dir = path.join( self.model_path, Params.Dataset['name'], f"{setname}Trio" )
                os.makedirs( out_dir, exist_ok=True )
                for data in tqdm(self.dataset[setname]):
                    sketch = data['imageA']
                    photo = data['imageB']

                    warpedSketch = self.model.warp(sketch, photo)

                    out_img.paste(sketch.resize(out_size), (0, 0))
                    out_img.paste(photo.resize(out_size), (out_size[0], 0))
                    out_img.paste(warpedSketch.resize(out_size), (out_size[0]*2, 0))
                    out_img.save( path.join(out_dir, data['fname']) )
