from PIL import Image

from models import get_model
from Params import Params
from Visualizer import Visualizer
from ShoesDataset import ShoesDataset
from SketchyDataset import SketchyDataset

import time
import os
import sys
import os.path as path
import torch
from tqdm import tqdm


class Tester:
    def __init__(self):
        self.visualizer = Visualizer()

        self.model_path = path.join( Params.CheckpointDir, Params.ExperimentName )

        self.load_dataset()
        self.create_model()
        self.load_latest_checkpoint()


    def load_dataset(self):
        self.test_set = ShoesDataset('test')

    def create_model(self):
        self.model = get_model(visualizer=self.visualizer)


    def load_latest_checkpoint(self):
        if path.exists( path.join(self.model_path, 'final.pt') ):
            latest_file = 'final.pt'
        else:
            # Find saved model with the latest epoch
            saved_models = [ (int(filename[6:-3]),filename) for filename in os.listdir(self.model_path) if filename.endswith('.pt') and filename.startswith('epoch_') ]
            saved_models.sort(key = lambda t: t[0], reverse=True)
            latest_file = saved_models[0][1]

        print( f"Loading model {latest_file}" )
        self.model.load( path.join(self.model_path, latest_file) )


    def run_test(self):
        print("Testing")

        out_dir = os.path.join( self.model_path, 'test_results' )
        os.makedirs( out_dir, exist_ok=True )

        out_size = (256, 256)
        out_img = Image.new('RGB', (out_size[0] * 3, out_size[1]))

        with torch.no_grad():
            self.model.prepare_testing()
            for data in tqdm(self.test_set):
                sketch = data['imageA']
                photo = data['imageB']

                warpedSketch = self.model.warp(sketch, photo)

                out_img.paste(sketch.resize(out_size), (0, 0))
                out_img.paste(photo.resize(out_size), (out_size[0], 0))
                out_img.paste(warpedSketch.resize(out_size), (out_size[0]*2, 0))
                out_img.save( path.join(out_dir, data['fname']) )


if __name__ == "__main__":
    t = Tester()
    t.run_test()
