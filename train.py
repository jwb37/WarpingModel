from models import get_model
from Params import Params
from Visualizer import Visualizer
from SketchyDataset import SketchyDataset

import time
import os
import sys
import os.path as path
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.visualizer = Visualizer()

        self.load_dataset()
        self.create_model()

        if Params.isTrue('ContinueTrain'):
            self.prepare_continued_training()
        else:
            self.start_epoch = 0


    def load_dataset(self):
        self.training_set = SketchyDataset('train')

        self.train_dl = torch.utils.data.DataLoader(
            self.training_set,
            batch_size = Params.BatchSize,
            shuffle = True,
            num_workers = 6
        )

    def prepare_logs(self):
        os.makedirs( path.join(Params.CheckpointDir, Params.ExperimentName), exist_ok=True )

        self.logs = {
            'train': path.join( Params.CheckpointDir, Params.ExperimentName, 'training.log' ),
        }

        # Do not clear log files if we are continuing training a previous model
        if not Params.isTrue('ContinueTrain'):
            for log_filename in self.logs.values():
                print( f"Clearing log file {log_filename}" )
                if path.exists(log_filename):
                    os.remove(log_filename)


    def prepare_continued_training(self):
        # Find latest model trained so far

        model_path = path.join( Params.CheckpointDir, Params.ExperimentName )
        # Thought about using a regular expression, but just a splice and startswith/endswith should suffice (it's not like we're expecting malformed filenames)
        saved_models = [ (int(filename[6:-3]),filename) for filename in os.listdir(model_path) if filename.endswith('.pt') and filename.startswith('epoch_') ]
        saved_models.sort(key = lambda t: t[0], reverse=True)

        print( f"Loading model at epoch {saved_models[0][0]}. Filename is {saved_models[0][1]}" )
        self.start_epoch = saved_models[0][0]
        self.model.load( path.join(model_path, saved_models[0][1]) )

        # Get Learning Rate scheduler up to appropriate point
        if Params.UseScheduler:
            for k in range(self.start_epoch):
                self.scheduler.step()


    def create_model(self):
        self.model = get_model(visualizer=self.visualizer)
        self.model.prepare_training()

        if Params.UseScheduler:
            self.scheduler = Params.create_scheduler(self.model.optimizer)

    def inner_training_loop(self, epoch):
        train_log_buffer = []

        print("Training")
        for iter, data in enumerate(tqdm(self.train_dl)):
            imageA = data['imageA'].cuda()
            imageB = data['imageB'].cuda()

            self.total_iter += imageA.size(0)
            self.visualizer.set_iter(epoch, self.total_iter)

            losses = self.model.training_step(imageA, imageB)

            loss_string = ' '.join( [f"L_{name} {value.item():.3f}" for name, value in losses.items()] )
            train_log_buffer.append(
                f"Epoch {epoch+1} Iter {iter} TotalIter {self.total_iter} {loss_string}\n"
            )

            if self.total_iter > self.next_losslog_iter:
                # Save training losses for one epoch in one go (keeps logs still valid in case of interruption mid-training)
                with open(self.logs['train'], 'a') as log_file:
                    log_file.writelines(train_log_buffer)

                train_log_buffer.clear()
                self.next_losslog_iter = self.total_iter + Params.LossLogFreq

            self.visualizer.save_images()


    def train(self):
        self.prepare_logs()

        # Initializes to zero, unless we're continuing training
        self.total_iter = self.start_epoch * len(self.training_set)
        self.visualizer.set_start_iter(self.total_iter)
        self.next_losslog_iter = self.total_iter + Params.LossLogFreq

#        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.start_epoch, Params.NumEpochs):

            print( f"\nEpoch {epoch+1} out of {Params.NumEpochs}:" )
            tic = time.perf_counter()

            self.inner_training_loop(epoch)

            if Params.UseScheduler:
                self.scheduler.step()

            # Save checkpoint
            if ((epoch+1) % Params.CheckpointFreq) == 0:
                filename = path.join(Params.CheckpointDir, Params.ExperimentName, f"epoch_{epoch+1}.pt")
                print( f"Saving checkpoint to {filename}" )
                self.model.save(filename)

            toc = time.perf_counter() - tic
            print( f"Epoch took {toc:.1f} seconds. Estimated remaining time {toc*(Params.NumEpochs - epoch - 1)/60:.1f} minutes" )

        # Save final model
        filename = path.join(Params.CheckpointDir, Params.ExperimentName, f"final.pt")
        self.model.save(filename)


if __name__ == "__main__":
    t = Trainer()
    t.train()
