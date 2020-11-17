import os
import time
import shutil
import argparse
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from dataset.shapenet import ShapeNet
from metric import Metric
from utils import *
from dataset.visualization import generate_curve


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
        model = get_model(config)
        self.model = model

        # Loss, Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(config['lr']),
                                          weight_decay=float(config['dr']))
        
        # Dataset and DataLoader
        self.train_set = ShapeNet(config, config_path, 'train')
        self.val_set = ShapeNet(config, config_path, 'valid')
        self.train_loader = DataLoader(self.train_set, batch_size=config['batch_size'],
                                       shuffle=True, num_workers=1, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=config['batch_size'],
                                     shuffle=True, num_workers=1, drop_last=False)

        print("Commencing training with {} model for {} epochs".format(config['model'],
                                                                       config['num_epochs']))
        print("Training set: {} samples".format(self.train_set.__len__()))
        print("Validation set: {} samples".format(self.val_set.__len__()))

        # Metrics
        self.metrics = Metric(config)
        self.train_metrics = MetricTracker(config, self.log_path, "train")
        self.val_metrics = MetricTracker(config, self.log_path, "val")
        self.epoch = 0

    def train(self):
        """Train the model over configured epochs and track
        metrics through training. Save the metrics once the training
        commences.
        """
        for self.epoch in range(int(self.config['num_epochs'])):
            # train and evaluate from an epoch
            self.run_epoch()
            # save model checkpoint
            self.save_model()

        # save scores across epochs
        self.train_metrics.save()
        self.val_metrics.save()

        # generate training / validation curves
        generate_curve([self.train_metrics.l2, self.val_metrics.l2],
                       ['Train', 'Valid'], "L2", self.exp_dir)
        generate_curve([self.train_metrics.iou, self.val_metrics.iou],
                       ['Train', 'Valid'], 'IoU', self.exp_dir)
        generate_curve([self.train_metrics.acc, self.val_metrics.acc],
                       ['Train', 'Valid'], 'Acc', self.exp_dir)
        generate_curve([self.train_metrics.loss, self.val_metrics.loss],
                       ['Train', 'Valid'], 'Loss', self.exp_dir)

    def run_epoch(self):
        """Training and evaluation loop for a given epoch.
        """
        # run a training epoch
        train_time = time.time()
        for i, sample in enumerate(self.train_loader, 0):
            inputs = sample['inputs']
            targets = sample['targets']

            # make prediction and compute loss
            preds = self.model(inputs)

            # compute loss, update parameters
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # track accuracy, IoU, and loss
            l2, iou, acc = self.get_metrics(preds, targets)
            self.train_metrics.store(l2, iou, acc, loss.item())
        train_time = time.time() - train_time

        # run an evaluation epoch
        self.model.eval()
        val_time = time.time()
        for i, sample in enumerate(self.val_loader, 0):
            inputs = sample['inputs']
            targets = sample['targets']

            # make prediction and compute loss
            preds = self.model(inputs)
            loss = self.criterion(preds, targets)

            # track accuracy, IoU, and loss
            l2, iou, acc = self.get_metrics(preds, targets)
            self.val_metrics.store(l2, iou, acc, loss.item())
        val_time = time.time() - val_time
        self.model.train()

        # average metrics over epoch and store
        self.train_metrics.store_epoch()
        self.val_metrics.store_epoch()

        ss = (train_time + val_time) / (self.train_set.__len__() + self.val_set.__len__())
        train_l2, train_iou, train_acc, train_loss = self.train_metrics.get_latest()
        val_l2, val_iou, val_acc, val_loss = self.val_metrics.get_latest()
        print(("epoch: {}/{}  |  t_l2: {:.3f}  |  t_iou: {:.3f}  |  t_acc: {:.3f}  |  t_loss: {:.3f}  |  " +
              "v_l2: {:.3f}  |  v_iou: {:.3f}  |  v_acc: {:.3f}  |  val_loss: {:.3f}  |  samples/sec: {:.2f}").
              format(self.epoch, self.config['num_epochs'], train_l2, train_iou, train_acc,
                     train_loss, val_l2, val_iou, val_acc, val_loss, ss))

    def get_metrics(self, preds, targets):
        """Compute batch accuracy and IoU
        """
        # compute L2 from DF
        preds = preds.detach().squeeze(1)
        targets = targets.detach().squeeze(1)
        l2 = self.metrics.l2_norm(preds, targets)

        # compute IoU and acc from thresholded Df
        preds = (preds < 0.50).long()
        targets = (targets < 0.50).long()
        iou = self.metrics.get_iou_per_batch(preds, targets)
        acc = self.metrics.get_accuracy_per_batch(preds, targets)
        return l2, iou, acc

    def save_model(self):
        """Save checkpoint of current model
        """
        model_path = os.path.join(self.model_path, get_model_name(self.config, self.epoch))
        torch.save(self.model.state_dict(), model_path)
        optim_path = os.path.join(self.model_path, 'optimizer')
        torch.save(self.optimizer.state_dict(), optim_path)


if __name__ == '__main__':
    # parse CLI inputs
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--pretrain', type=bool, default=False, help="Continue training a previous model")
    args = parser.parse_args()

    # create trainer and start training
    trainer = Trainer(args)
    trainer.train()
