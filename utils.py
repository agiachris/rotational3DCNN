import os
import yaml
import torch
import numpy as np
from collections import Counter
from models import UNet3dBaseline
from dataset.visualization import generate_curve


def get_config(path, config_file):
    """Load YAML configuration file
    """
    with open(os.path.join(path, config_file), 'r') as config_stream:
        config = yaml.safe_load(config_stream)
    return config


def save_config(path, config):
    """Save YAML configuration file
    """
    path = os.path.join(path, 'config.yaml')
    with open(path, 'w') as config_stream:
        yaml.dump(config, config_stream)


def get_model(config):
    """Return the model specified in the configuration object
    """
    if config['model'] == "baseline":
        num_filters = int(config['num_filters'])
        return UNet3dBaseline.UNet3dBaseline(1, 1, num_filters)


def get_model_name(config, epoch):
    """Create a model name based on experiment detail
    for checkpoint saving.
    :param config: experiment configuration object
    :param epoch: current training epoch
    :return: model name
    """
    model_name = "{}_bsz{}_lr{}_dr{}_epo{}".format(
        config['model'],
        config['batch_size'],
        config['lr'],
        config['dr'],
        epoch + 1
    )
    return model_name


class MetricTracker:

    def __init__(self, config, save_path, name):
        """
        Metric tracking class. Provides functionality for tracking, storing,
        and saving metrics over an entire training session.
        :param config: Experiment configuration file
        :param save_path: Path to save final results
        :param name: Identifier for the data split (e.g. 'train', 'val', 'test')
        """
        self.name = name
        self.save_path = os.path.join(save_path, name)
        self.l2 = np.zeros(config['num_epochs'])
        self.iou = np.zeros(config["num_epochs"])
        self.acc = np.zeros(config["num_epochs"])
        self.loss = np.zeros(config["num_epochs"])
        self.tracker = Counter()
        self.idx = 0

    def store_epoch(self):
        """Average the metrics over the number of training
        iterations in a given epoch, and store.
        """
        total_epoch = self.tracker["i"]
        for metric in self.tracker.keys():
            self.tracker[metric] /= total_epoch
        self.l2[self.idx] = self.tracker["l2"]
        self.iou[self.idx] = self.tracker["iou"]
        self.acc[self.idx] = self.tracker["acc"]
        self.loss[self.idx] = self.tracker["loss"]
        self.reset()
        self.idx += 1

    def store(self, l2, iou, acc, loss):
        """Store metrics for a given batch.
        """
        self.tracker["i"] += 1
        self.tracker["l2"] += l2
        self.tracker["iou"] += iou
        self.tracker["acc"] += acc
        self.tracker["loss"] += loss

    def get_latest(self):
        """Return the most recent performance metrics
        """
        l2 = self.l2[self.idx-1]
        iou = self.iou[self.idx-1]
        acc = self.acc[self.idx-1]
        loss = self.loss[self.idx-1]
        return l2, iou, acc, loss

    def reset(self):
        """Reset tracker for a new epoch.
        """
        self.tracker = Counter()

    def save(self):
        """Save metrics over the entire session.
        """
        np.savetxt(self.save_path + '_l2', self.l2)
        np.savetxt(self.save_path + "_iou", self.iou)
        np.savetxt(self.save_path + "_acc", self.acc)
        np.savetxt(self.save_path + "_loss", self.loss)
