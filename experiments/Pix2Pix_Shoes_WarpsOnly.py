import torch

CheckpointFreq = 10

ExperimentName = 'Pix2Pix (Mimic Shoes Warpsonly)'
ModelName = 'Pix2Pix'
InputNC = 1
OutputNC = 3

Dataset = {
    'name': 'Shoes',
    # Use overwrite_dir to use results from another model in training this model
    'overwrite_dir': 'checkpoints/Mimic (RTN Shoes)/Shoes',
    'A_suffixes': ['Warped']
}
Device = 'cuda'

CropSize = 256
BatchSize = 1
NumEpochs = 150

LossLogFreq = 1000
VisualizerFreq = 200
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False