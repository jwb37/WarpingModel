import torch

CheckpointFreq = 50
ExperimentName = 'Pix2Pix (Mimic Chairs, WarpsOnly)'

ModelName = 'Pix2Pix'
InputNC = 1
OutputNC = 3

Dataset = {
    'name': 'Chairs',
    # Use overwrite_dir to use results from another model in training this model
    'overwrite_dir': 'checkpoints/Mimic (RTN Chairs)/Chairs',
    'A_suffixes': ['Warped']
}
Device = 'cuda'

CropSize = 256
BatchSize = 1
NumEpochs = 450

LossLogFreq = 1000
VisualizerFreq = 1500
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
