import torch

CheckpointFreq = 10

ExperimentName = 'Mimic (RTN Shoes, Blur)'
ModelName = 'Mimic'
Dataset = {
    'name': 'Shoes',
    # Use overwrite_dir to use results from another model in training this model
    'overwrite_dir': 'checkpoints/RTN_Adv_Shoes/Shoes (blur)'
}
Device = 'cuda'

CropSize = 256
BatchSize = 16
NumEpochs = 40

LossLogFreq = 1000
VisualizerFreq = 500
VisualizerNumExemplars = 1

# Learning Rate
create_optimizer = lambda params: torch.optim.Adam(params, lr=1e-5, betas=(0.5, 0.999))
UseScheduler = False
