import torch

CheckpointDir = './checkpoints'
ExperimentName = 'MimicChairs'

ModelName = 'Mimic'
Dataset = 'Chairs'
Device = 'cuda'

CropSize = 256
NumWarpIterations = 3

create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
