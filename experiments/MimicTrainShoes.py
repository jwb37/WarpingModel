import torch

CheckpointDir = './checkpoints'
CheckpointFreq = 1

ExperimentName = 'Mimic'
ModelName = 'Mimic'
Dataset = 'Shoes'
B_suffix = 'Flow'
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 40
NumWarpIterations = 3
#ContinueTrain = True

LossLogFreq = 1000
VisualizerFreq = 500
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, NumEpochs // 3, gamma=0.1)