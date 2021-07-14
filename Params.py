import torch

CheckpointDir = './checkpoints'
CheckpointFreq = 1

ModelName = 'ImageWarpNet'
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 40
#ContinueTrain = True

LossLogFreq = 100

TrainingSet = './Sketchy/all'
TestSet = './Sketchy/test'

# Learning Rate
create_optimizer = lambda params: torch.optim.SGD(params, lr=0.02) #, betas=(0.5, 0.999))
UseScheduler = False
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, NumEpochs // 3, gamma=0.1)

def isTrue(var_name):
    if not hasattr(__name__,var_name):
        return False
    else:
        return getattr(__name__,var_name)
