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
        osj = os.path.join

        base_path = os.path.dirname(os.path.abspath(__file__))
        config = get_config(base_path, args.config)
        self.config = config
        config_path = os.path.split(osj(base_path, args.config))[0]

        # Model - build in pre-trained load
        model = get_model(config)
        self.model = model

        # Loss, Optimizer
        self.criterion = nn.BCEWithLogitsLoss()

        # Dataset and DataLoader
        self.dataset = ShapeNet(config, config_path, args.type)

        self.loader = DataLoader(self.dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=1, drop_last=False)

        # Metrics
        self.metrics = Metric(config)
        self.epoch = 0

    def eval(self):
        for i, sample in enumerate(self.loader, 0):
            inputs = sample['inputs']
            targets = sample['targets']
            out = self.model(inputs)
            loss = self.criterion(out, targets) # compute the loss
            acc, iou = self.get_metrics(out.detach().squeeze(1), targets.detach().squeeze(1))

            if i % 10 == 0:
                print("batch {} of {}".format(i, len(self.loader)))            
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
    parser.add_argument('--type', required=True, help='Specify whether we are evaluating training or validation data')

    args = parser.parse_args()

    # create trainer and start training
    evaluate = Evaluater(args)
    evaluate.eval()