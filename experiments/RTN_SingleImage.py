import torch

CheckpointFreq = 1

ExperimentName = 'RTN_Adv_Shoes'
ModelName = 'Affine'
Dataset = {
    'name': 'Single'
}
Device = 'cuda'

CropSize = 256
NumWarpIterations = 4
TrainVGG = True

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
