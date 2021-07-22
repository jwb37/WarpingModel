import torch

CheckpointDir = './checkpoints'
CheckpointFreq = 1

ModelName = 'ImageWarpNet'
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 40
NumWarpIterations = 0
#ContinueTrain = True

LossLogFreq = 1000
VisualizerFreq = 1000
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.SGD(params, lr=0.02) #, betas=(0.5, 0.999))
UseScheduler = False
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, NumEpochs // 3, gamma=0.1)

# Loss function
loss = {
    'type': 'LSeSim',
    'num_patches': 64,
    'patch_size': 9,
    'use_attn': True,
    'use_norm': True,
    'ssim_compare_fn': 'cos',
    'T': 0.1,
    'attn_init_info': {
        'init_type': 'normal',
        'init_gain': 0.02,
        'gpu_ids': []
    }
}

TrainingSet = './Sketchy/all'
TestSet = './Sketchy/test'


def isTrue(var_name):
    if not hasattr(__name__,var_name):
        return False
    else:
        return getattr(__name__,var_name)
