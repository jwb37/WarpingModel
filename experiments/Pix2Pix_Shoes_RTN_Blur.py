import torch

CheckpointFreq = 130

ExperimentName = 'Pix2Pix (RTN Shoes, Blur)'
ModelName = 'Pix2Pix'
InputNC = 2
OutputNC = 3

Dataset = {
    'name': 'Shoes',
    # Use overwrite_dir to use results from another model in training this model
    'overwrite_dir': 'checkpoints/RTN_Shoes/Shoes(blur)',
    'A_suffixes': ['A','Warped']
}
Device = 'cuda'

CropSize = 256
BatchSize = 1
NumEpochs = 150

LossLogFreq = 1000
VisualizerFreq = 1500
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
