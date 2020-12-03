import os
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import utils
from dataset.data_utils import *
from torch.utils.data import Dataset

def compare_experiments(names, paths, metrics, save_path):
    """Generate plots comparing desired metrics across given models.
    args:
        :names: Names of the experiment corresponding to paths
        :paths: List of paths to the experiment directories
        :metrics: Desired metrics to generate plots of
        :save_path: path to save the plots
        :epochs: the number of data samples to plot
    """
    save_name = 'compare'
    for name in names:
        save_name += '_' + name
    save_dir = os.path.join(save_path, save_name)
    os.makedirs(save_dir)

    # load data from experiment directories
    data = {}
    for name, path in zip(names, paths):
        if name not in data:
            data[name] = {}
        for metric in metrics:
            data[name][metric] = {}
            for split in ['train', 'val']:
                path_to_load = os.path.join(path, '{}_{}'.format(split, metric))
                data[name][metric][split] = np.loadtxt(path_to_load, delimiter=',')
                epochs = len(data[name][metric][split])

    # generate and save plots
    for metric in metrics:
        fig = plt.figure()
        plt.title("{} Model Comparison".format(metric.capitalize()))
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        for name in names:
            for split in ['train', 'val']:
                label = name + "_" + split
                plt.plot(range(1, epochs+1), data[name][metric][split][:epochs], label=label)

        plt.legend(loc='best')
        plt.savefig(os.path.join(save_dir, '{}_curve.png'.format(metric)))
        plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Comparison')
    parser.add_argument('--model1', required=True, help='Path to first model')
    parser.add_argument('--model2', required=True, help='Path to second model')
    args = parser.parse_args()

    # Used specifically for generating metric curves across experiments
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = ['l2', 'iou', 'acc', 'loss']
    names = [args.model1, args.model2]
    path1 = base_path + '/rotational3DCNN/exp/' + names[0] + '/logs'
    path2 = base_path + '/rotational3DCNN/exp/' + names[1] + '/logs'
    paths = [path1, path2]
    print(paths)
    compare_experiments(names, paths, metrics, os.path.join(base_path, 'rotational3DCNN/exp'))
