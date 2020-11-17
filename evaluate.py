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


class Evaluater:

    def __init__(self, args):
        """
        Create Trainer object to which handles the training and evaluation of a specified model,
        the tracking of computed metrics, and saving of results / checkpoints.
        :param args: ArgParse object which holds the path the experiment configuration file along
                     with other key experiment options.
        """
        # get experiment configuration file
        base_path = os.path.dirname(os.path.abspath(__file__))
        config = get_config(base_path, args.config)
        self.config = config

        # Model - build in pre-trained load
        model = get_model(config)
        self.model = model

        # Loss, Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=config['lr'],
                                          weight_decay=config['dr'])

        # Dataset and DataLoader
        self.dataset = ShapeNet(config)
        self.loader, self.labels = DataLoader(self.dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=1, drop_last=False)

        # Metrics
        self.metrics = Metric()
        self.train_metrics = MetricTracker(config, self.exp_dir, "train")
        self.val_metrics = MetricTracker(config, self.exp_dir, "val")
        self.epoch = 0

    def eval(self):
        out = self.model(self.loader)
        loss = self.criterion(out, self.labels) # compute the loss
        acc, iou = self.get_metrics(out,self.labels)
        print("Accuracy:",acc,"IOU:",iou,"Loss:",loss)

    def get_metrics(self, preds, labels):
        """Compute batch accuracy and IoU
        """
        acc = self.metrics.get_accuracy_per_batch(preds, labels)
        iou = self.metrics.get_iou_per_batch(preds, labels)
        return acc, iou


if __name__ == '__main__':
    # parse CLI inputs
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--pretrain', type=bool, default=False, help="Continue training a previous model")
    args = parser.parse_args()

    # create trainer and start training
    evaluate = Evaluater(args)
    evaluate.eval()