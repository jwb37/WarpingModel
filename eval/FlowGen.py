import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from Params import Params
from .BaseEval import BaseEval
from dataset_classes import create_dataset

import os

class FlowGen(BaseEval):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        self.dataset = {
            'train': create_dataset('train', ret_tensor=True),
            'test': create_dataset('test', ret_tensor=True)
        }

    def run_test(self):
        print("Running test...")

        with torch.no_grad():
            self.model.prepare_testing()
            for setname in ('train','test'):
                print(f"Generating flows for {setname} dataset...")
                out_dir = os.path.join( self.model_path, Params.Dataset['name'], setname + 'Flow' )
                os.makedirs( out_dir, exist_ok=True )
                for data in tqdm(self.dataset[setname]):
                    sketch = data['imageA'].unsqueeze(0).to(Params.Device)
                    photo = data['imageB'].unsqueeze(0).to(Params.Device)

                    flow = self.model.calc_flow(sketch, photo)
                    if Params.isTrue('BlurFlow'):
                        flow = F.gaussian_blur(flow, kernel_size=7)
                    torch.save( flow.cpu(), os.path.join(out_dir, data['fname'][:-3] + 'pt') )
