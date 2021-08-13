import torch
from tqdm import tqdm

from .BaseEval import BaseEval 
from dataset_classes import create_dataset
from Params import Params

import os

class WarpedGen(BaseEval):
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

        with torch.no_grad():
            self.model.prepare_testing()
            for setname in ('train', 'test'):
                out_dir = os.path.join( self.model_path, Params.Dataset, setname + 'Warped' )
                os.makedirs( out_dir, exist_ok=True )
                for data in tqdm(self.dataset[setname]):
                    sketch = data['imageA']
                    photo = data['imageB']

                    warpedSketch = self.model.warp(sketch, photo)
                    warpedSketch.resize(out_size).save( os.path.join(out_dir, data['fname']) )
