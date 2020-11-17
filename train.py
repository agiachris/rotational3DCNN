import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from dataset.shapenet import ShapeNet
from metric import Metric
from utils import *


class Trainer:

    def __init__(self, args):
        """
        Create Trainer object to which handles the training and evaluation of a specified model,
        the tracking of computed metrics, and saving of results / checkpoints.

        :param args: ArgParse object which holds the path the experiment configuration file along
                     with other key experiment options.
        """
        osj = os.path.join

        # get experiment configuration file
        base_path = os.path.dirname(os.path.abspath(__file__))
        config = get_config(base_path, args.config)
        self.config = config
        config_path = os.path.split(osj(base_path, args.config))[0]

        # create experiment, logging, and model checkpoint directories
        self.exp_dir = osj(osj(base_path, config['exp_dir']), config['exp_name'])
        self.log_path = osj(self.exp_dir, "logs")
        self.model_path = osj(self.exp_dir, "checkpoints")
        if os.path.exists(self.exp_dir):
            print("Error: This experiment already exists")
            print("Cancel Session:    0")
            print("Overwrite:         1")
            if input() == str(1):
                shutil.rmtree(self.exp_dir)
            else:
                exit(1)
        os.mkdir(self.exp_dir)
        os.mkdir(self.log_path)
        os.mkdir(self.model_path)

        # Model - build in pre-trained load
        # model = get_model(config)
        # self.model = model

        # Loss, Optimizer
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.optimizer = torch.optim.Adam(model.parameters(),
        #                                   lr=config['lr'],
        #                                   weight_decay=config['dr'])

        # Dataset and DataLoader
        self.train_set = ShapeNet(config, config_path, 'train')
        self.val_set = ShapeNet(config, config_path, 'valid')
        self.train_loader = DataLoader(self.train_set, batch_size=config['batch_size'],
                                       shuffle=True, num_workers=1, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=config['batch_size'],
                                     shuffle=True, num_workers=1, drop_last=False)

        # Metrics
        self.metrics = Metric()
        self.train_metrics = MetricTracker(config, self.exp_dir, "train")
        self.val_metrics = MetricTracker(config, self.exp_dir, "val")
        self.epoch = 0

    def train(self):
        """Train the model over configured epochs and track
        metrics through training. Save the metrics once the training
        commences.
        """
        for self.epoch in self.config['num_epochs']:
            # train and evaluate from an epoch
            self.run_epoch()

            # average metrics over epoch and store
            self.train_metrics.store_epoch()
            self.val_metrics.store_epoch()

            # save model checkpoint
            self.save_model()

        # save scores across epochs
        self.train_metrics.save()
        self.val_metrics.save()

    def run_epoch(self):
        """Training and evaluation loop for a given epoch.
        """
        # run a training epoch

        self.model.eval()

        # store batch metrics with: self.train_metrics.store(acc, iou, loss)

        # run an evaluation epoch

        self.model.train()

    def get_metrics(self, preds, labels):
        """Compute batch accuracy and IoU
        """
        acc = self.metrics.get_accuracy_per_batch(preds, labels)
        iou = self.metrics.get_iou_per_batch(preds, labels)
        return acc, iou

    def save_model(self):
        """Save checkpoint of current model
        """
        model_path = os.path.join(self.model_path, get_model_name(self.config, self.epoch))
        torch.save(self.model.state_dict(), model_path)
        optim_path = os.path.join(self.model_path, 'optimizer')
        torch.save(self.optimizer.state_dict(), optim_path)

    def load_model(self):
        """Load a model and optimizer from checkpoint
        """
        # TODO
        return


if __name__ == '__main__':
    # parse CLI inputs
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--pretrain', type=bool, default=False, help="Continue training a previous model")
    args = parser.parse_args()

    # create trainer and start training
    trainer = Trainer(args)
    trainer.train()
