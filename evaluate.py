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
from dataset.visualization import *


class Evaluator:

    def __init__(self, args):
        """
        Create Evaluator object which handles the evaluation of a specified model,
        the tracking of computed metrics, and saving of results.

        :param args: ArgParse object which holds the path the experiment configuration file along
                     with other key experiment options.
        """
        # get experiment configuration file
        osj = os.path.join
        oss = os.path.split

        base_path = os.path.dirname(os.path.abspath(__file__))
        config = get_config(base_path, args.config)
        self.config = config
        config_path = osj(base_path, 'config')

        # create experiment, logging, and model checkpoint directories
        model_name = os.path.split(args.model)[1].split('.')[0]
        self.eval_dir = osj(base_path, oss(args.config)[0], 'eval_' + args.data + '_' + model_name)
        self.visual_path = osj(self.eval_dir, 'visuals')
        if os.path.exists(self.eval_dir):
            print("Error: This evaluation already exists")
            print("Cancel Session:    0")
            print("Overwrite:         1")
            if input() == str(1):
                shutil.rmtree(self.eval_dir)
            else:
                exit(1)
        os.mkdir(self.eval_dir)
        os.mkdir(self.visual_path)

        # Device
        self.device = torch.device("cpu" if args.gpuid == 'cpu' else "cuda:{}".format(args.gpuid))

        # Model - load pre-trained
        model = get_model(config)
        self.model = model
        self.load_model(args.model)
        self.model.to(self.device)

        # Loss metric
        self.criterion = nn.MSELoss()

        # Dataset and DataLoader
        self.data_type = args.data
        self.dataset = ShapeNet(config, config_path, args.data)
        self.tracked_samples = extract_categories(self.dataset.samples, 2)
        self.loader = DataLoader(self.dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=1, drop_last=False)

        print("Commencing evaluation with {} model on {} split".format(config['model'], args.data))

        # Metrics
        self.metrics = Metric(config)
        self.metric_tracker = MetricTracker(config, self.eval_dir, args.data)
        self.epoch = 0

    def eval(self):
        self.model.eval()
        val_time = time.time()
        for i, sample in enumerate(self.loader, 0):
            inputs = sample['inputs'].to(self.device)
            targets = sample['targets'].to(self.device)

            # make prediction and compute loss
            preds = self.model(inputs)
            loss = self.criterion(preds, targets)

            # track accuracy, IoU, and loss
            l2, iou, acc = self.get_metrics(preds.detach().cpu(), targets.detach().cpu())
            self.metric_tracker.store(l2, iou, acc, loss.detach().cpu().item())

        for item_class, item_list in self.tracked_samples.items():
            for i, pair in enumerate(item_list):
                generate_original_voxel_image(self.data_type, pair[1], i,
                                                class_mapping[str(item_class)], self.visual_path)
                generate_voxel_image_from_model(self.data_type, self.model, pair[0], i,
                                                class_mapping[str(item_class)], self.epoch,
                                                self.visual_path, self.device)

        # average metrics over epoch
        ss = (time.time() - val_time) / (self.dataset.__len__())
        self.metric_tracker.store_epoch()
        l2, iou, acc, loss = self.metric_tracker.get_latest()
        print("Evaluation results - l2: {:.3f}  |  iou: {:.3f}  |  acc: {:.3f}  |  loss: {:.3f} |   sec/sample: {:.2f}"
              .format(l2, iou, acc, loss, ss))
        self.metric_tracker.save()

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

    def load_model(self, model_path):
        """Load state dictionary of model
        """
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)


if __name__ == '__main__':
    # parse CLI inputs
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--model', required=True, help='Model version to evaluate')
    parser.add_argument('--data', required=True, help='Data split to evaluate the model on')
    parser.add_argument('--gpuid', type=str, default='cpu', help="Training device")
    args = parser.parse_args()

    # create evaluator and start evaluating
    evaluate = Evaluator(args)
    evaluate.eval()