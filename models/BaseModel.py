import torch
from itertools import chain

class BaseModel:
    def __init__(self, visualizer=None):
        self.visualizer = visualizer
        self.trained_nets = dict()
        self.optimizers = dict()

    #---------------------------------------------------------------
    # Training functions

    def prepare_training(self):
        for net in self.trained_nets.values():
            net.train()

    def prepare_testing(self):
        for net in self.trained_nets.values():
            net.eval()

    def training_step(self, tensorA, tensorB):
        pass

    #-----------------------------------------------------
    def save(self, filename):
        save_state = {
            name: net.state_dict()
            for name, net in chain(self.trained_nets.items(), self.optimizers.items())
        }
        torch.save( save_state, filename )

    def load(self, filename):
        save_state = torch.load( filename )
        for name, net in self.trained_nets.items():
            if name in save_state:
                self.trained_nets[name].load_state_dict(save_state[name])
            else:
                print(f"Warning: no saved parameters found for network {name}")

        for name, net in self.optimizers.items():
            if name in save_state:
                self.optimizers[name].load_state_dict(save_state[name])
            else:
                print(f"Warning: no saved parameters found for optimizer {name}")
