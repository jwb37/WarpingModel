import torch
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from .BaseEval import BaseEval
from dataset_classes import create_dataset

import os
import os.path as path

from Params import Params


class I2I(BaseEval):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        self.dataset = create_dataset('test', ret_tensor=True)

    def run_test(self):
        print("Testing")

        out_size = (256, 256)

        with torch.no_grad():
            self.model.prepare_testing()
            out_dir = path.join( self.model_path, Params.Dataset['name'], f"test_results" )
            os.makedirs( out_dir, exist_ok=True )
            for data in tqdm(self.dataset):
                sketch = data['imageA'].unsqueeze(dim=0)
                tens = self.model(sketch)

                tens = tens.detach().cpu().squeeze(dim=0)
                tens = (tens + 1) / 2.0
                fake_photo = F.to_pil_image(tens)

                fake_photo.save( path.join(out_dir, data['fname']) )
