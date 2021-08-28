import torch

CheckpointFreq = 5

ExperimentName = 'RTN_Chairs'
ModelName = 'Affine'
Dataset = {
    'name': 'Chairs'
}
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 20
NumWarpIterations = 4
TrainVGG = True

LossLogFreq = 1000
VisualizerFreq = 500
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
