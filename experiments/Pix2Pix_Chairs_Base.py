import torch

CheckpointFreq = 60

ExperimentName = 'Pix2Pix (Base Chairs)'
ModelName = 'Pix2Pix'
InputNC = 1
OutputNC = 3

Dataset = {
    'name': 'Chairs'
}
Device = 'cuda'

CropSize = 256
BatchSize = 1
NumEpochs = 600

LossLogFreq = 1000
VisualizerFreq = 1500
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
