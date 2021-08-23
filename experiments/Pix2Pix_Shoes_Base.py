import torch

CheckpointFreq = 10

ExperimentName = 'Pix2Pix (Base)'
ModelName = 'Pix2Pix'
InputNC = 1
OutputNC = 3

Dataset = {
    'name': 'Shoes'
}
Device = 'cuda'

CropSize = 256
BatchSize = 1
NumEpochs = 150

LossLogFreq = 1000
VisualizerFreq = 200
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
