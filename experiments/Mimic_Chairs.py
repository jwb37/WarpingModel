import torch

CheckpointFreq = 40

ExperimentName = 'Mimic (RTN Chairs)'
ModelName = 'Mimic'
Dataset = {
    'name': 'Chairs',
    # Use overwrite_dir to use results from another model in training this model
    'overwrite_dir': 'checkpoints/RTN_Chairs/Chairs'
}
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 120

LossLogFreq = 1000
VisualizerFreq = 500
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
