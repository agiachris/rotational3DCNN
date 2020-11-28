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

        base_path = os.path.dirname(os.path.abspath(__file__))
        config = get_config(base_path, args.config)
        self.config = config
        config_path = os.path.split(osj(base_path, args.config))[0]

        # create experiment, logging, and model checkpoint directories
        self.eval_dir = osj(osj(base_path, config['eval_dir']), config['eval_name'])
        self.visual_path = osj(self.eval_dir, "visuals")
        if os.path.exists(self.eval_dir):
            print("Error: This evaluation already exists")
            print("Cancel Session:    0")
            print("Overwrite:         1")
            if input() == str(1):
                shutil.rmtree(self.exp_dir)
            else:
                exit(1)
        os.mkdir(self.eval_dir)
        os.mkdir(self.visual_path)

        # Device
        self.device = torch.device("cpu" if args.gpuid=='cpu' else "cuda:{}".format(args.gpuid))

        # Model - build in pre-trained load
        model = get_model(config)
        self.model = model.to(self.device)

        # Loss, Optimizer
        self.criterion = nn.BCEWithLogitsLoss()

        # Dataset and DataLoader
        self.dataset = ShapeNet(config, config_path, args.type)
        self.tracked_samples = extract_categories(self.dataset.samples, 3)

        self.loader = DataLoader(self.dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=1, drop_last=False)

        # Metrics
        self.metrics = Metric(config)
        self.epoch = 0

    def eval(self):
        for i, sample in enumerate(self.loader, 0):
            inputs = sample['inputs'].to(self.device)
            targets = sample['targets'].to(self.device)
            out = self.model(inputs)
            loss = self.criterion(out, targets) # compute the loss
            acc, iou = self.get_metrics(out.detach().squeeze(1), targets.detach().squeeze(1))

            if i % 10 == 0:
                print("batch {} of {}".format(i, len(self.loader)))            
                print("Accuracy:",acc,"IOU:",iou,"Loss:",loss)
            for item_class, item_list in self.tracked_samples.items():
                for i, pair in enumerate(item_list):
                    generate_original_voxel_image(pair[1], i, class_mapping[str(item_class)], self.visual_path)
                    generate_voxel_image_from_model(self.model, pair[0], i, class_mapping[str(item_class)], self.epoch, self.visual_path, self.device)

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
    parser.add_argument('--gpuid', type=str, default='cpu', help="Training device")

    args = parser.parse_args()

    # create evaluator and start evaluating
    evaluate = Evaluator(args)
    evaluate.eval()