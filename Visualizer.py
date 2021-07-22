from PIL import Image
import torch
import numpy as np

import os
import random

import Params


class Visualizer:
    def __init__(self):
        self.img_batch = dict()
        self.next_save_iter = Params.VisualizerFreq
        self.save_this_iter = False
        self.save_path = os.path.join( Params.CheckpointDir, Params.ModelName, 'Images' )
        os.makedirs( self.save_path, exist_ok=True )


    def set_iter(self, epoch, total_iters):
        self.epoch = epoch
        self.iter = total_iters
        if total_iters >= self.next_save_iter:
            self.img_batch.clear()
            self.next_save_iter += Params.VisualizerFreq
            self.save_this_iter = True
            self.batch_indices = random.sample(range(Params.BatchSize), k=Params.VisualizerNumExemplars)
        else:
            self.save_this_iter = False


    def add_tensor(self, name, tensor):
        if not self.save_this_iter:
            return

        batch_size, C, H, W = tensor.size()
        if C == 3:
            img_format = 'RGB'
        else:
            img_format = 'L'

        imgs = []

        for n in self.batch_indices:
            img_np = tensor[n].detach().cpu().numpy()
            img_np = img_np.transpose( (2,1,0) )   # CxHxW -> WxHxC
            if C != 3:
                img_np = np.mean(img_np, -1, keepdims=False)     # Reduce non-rgb tensors to greyscale by taking mean activation value
                img_np = normalize_image(img_np)

            img = Image.fromarray(img_np, mode=img_format)
            imgs.append(img)

        self.img_batch[name] = imgs


    def save_images(self):
        if not self.save_this_iter:
            return

        for img_name, img_list in self.img_batch.items():
            for batch_idx, img in enumerate(img_list):
                fname = f"epoch({self.epoch})_iter({self.iter})_{img_name}_{batch_idx}.png"
                img.save( os.path.join(self.save_path, fname) )


def normalize_image(img):
    minval = img.min()
    maxval = img.max()
    eps = 1e-8

    return (img - minval) * 255 / (maxval - minval + eps)
