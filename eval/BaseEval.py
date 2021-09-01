from Visualizer import Visualizer
from models import get_model

import torch
from tqdm import tqdm

import os
import sys
import os.path as path

from Params import Params


class BaseEval:
    def __init__(self):
        self.model_path = path.join( Params.CheckpointDir, Params.ExperimentName )

        self.create_model()
        self.load_latest_checkpoint()


    def create_model(self):
        self.model = get_model()


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


    def eval(self):
        Params.isTrain = False
        self.load_dataset()
        self.model.prepare_testing()   
        self.run_test()

    # ----------------------------------------------
    # Abstract methods

    def load_dataset(self):
        pass

    def run_test(self):
        pass
