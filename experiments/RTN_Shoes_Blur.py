import torch

CheckpointFreq = 5

ExperimentName = 'RTN_Shoes'
ModelName = 'Affine'
Dataset = {
    'name': 'Shoes'
}
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 15
NumWarpIterations = 4
TrainVGG = True
BlurFlow = True

LossLogFreq = 1000
VisualizerFreq = 5000
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
