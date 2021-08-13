import torch
ExperimentName = 'RTN'

CheckpointDir = './checkpoints'

ModelName = 'Affine'
Dataset = 'Chairs'
Device = 'cuda'

CropSize = 256
NumWarpIterations = 3

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
