import os
import argparse
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from data.shapenet import ShapeNet
from utils import *


class Trainer:

    def __init__(self, args):

        # directories
        base_dir = os.path.realpath(args.config)

        # acquire configuration file
        config = get_config(args.config)
        self.config = config

        # setup saving operations

        # load model (pre-trained or saved) - have a model configuration
        self.model = get_model(config)

        # load datasets - create dataloaders (use config file)
        self.dataset = ShapeNet(config)


        # setup metric global metric structures (loss, metric classes)
        self.criterion = nn.BCEWithLogitsLoss()


    def train(self):
        self.epoch = 0

        # loop over epochs
        for self.epoch in self.config['num_epochs']:
            self.run_epoch()
            self.save()

        self.save_metrics()
        return 0

    def run_epoch(self):

        # run a training epoch

            # track metrics

        # model.eval()
        # run an evaluation epoch

            # track metrics

    def get_metrics(self, preds, labels):
        # compute desired metrics for batch and return
        return

    def save(self):
        # save current model instance
        return


if __name__ == '__main__':
    # parse CLI inputs
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--continue', type=bool, default=False, help="Continue training a previous model")
    args = parser.parse_args()

    # create trainer and start training
    trainer = Trainer(args)
    trainer.train()
