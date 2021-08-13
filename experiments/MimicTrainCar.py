import torch

CheckpointDir = './checkpoints'
CheckpointFreq = 10

ExperimentName = 'MimicCar'
ModelName = 'Mimic'
Dataset = 'SketchyCOCO'
SketchyCOCO_Categories = [3]
B_subfolder = 'Flow'
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 120
NumWarpIterations = 3
#ContinueTrain = True

LossLogFreq = 1000
VisualizerFreq = 500
VisualizerNumExemplars = 1


create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
