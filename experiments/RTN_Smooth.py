import torch

CheckpointDir = './checkpoints'
CheckpointFreq = 1

ExperimentName = 'RTN_Smooth'
ModelName = 'Affine'
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 20
NumWarpIterations = 3
#ContinueTrain = True'

LossLogFreq = 300
VisualizerFreq = 500
VisualizerNumExemplars = 1

UseSmoothLoss = True
SmoothLossLambda = 0.001

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, NumEpochs // 3, gamma=0.1)


TrainingSet = './Sketchy/all'
TestSet = './Sketchy/test'

