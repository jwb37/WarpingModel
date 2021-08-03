import torch

CheckpointDir = './checkpoints'
CheckpointFreq = 1

ExperimentName = 'WST'
ModelName = 'Translation'
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 20
NumWarpIterations = 2
#ContinueTrain = True

LossLogFreq = 1000
VisualizerFreq = 1000
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))
UseScheduler = False
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, NumEpochs // 3, gamma=0.1)

# Loss function
loss = {
    'type': 'wst',
}

TrainingSet = './Sketchy/all'
TestSet = './Sketchy/test'


def isTrue(var_name):
    if not hasattr(__name__,var_name):
        return False
    else:
        return getattr(__name__,var_name)
