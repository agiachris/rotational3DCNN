import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from utils import *
from torch.utils.data import DataLoader
from dataset.data_utils import *
from torch.utils.data import Dataset


def visualize_df(tensor):
    """Basic 3D visualization for a distance field tensor.

    :param tensor: Distance Field (DF) tensor, np.array, shape = (32, 32, 32)
    :return: None
    """

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    # The following portion attempts to extract the shape from the DF
    # display zero voxels in DF
    x, y, z = np.where(tensor <= 0.5)
    color = tensor[x, y, z]

    fig2 = plt.figure("Zero DF")
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(z, x, y, zdir='z', c=cm.jet(color), s=0.6)


def visualize_sdf(tensor):
    """Basic 3D visualization for a signed distance field tensor.

    :param tensor: Signed Distance Field (SDF) tensor, np.array, shape = (32, 32, 32)
    :return: None
    """

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    # find voxels with finite sdf values
    finite_region = np.isfinite(tensor) * 1
    x, y, z = np.nonzero(finite_region)
    finite_color = tensor[x, y, z]

    # color mapping for finite regions
    norm = matplotlib.colors.Normalize(vmin=min(finite_color), vmax=max(finite_color))
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # plot finite SDF voxels
    fig = plt.figure("Finite SDF")
    ax = fig.add_subplot(111, projection='3d')
    color_list = []
    for i in range(len(x)):
        color_list.append(m.to_rgba(finite_color[i]))
    ax.scatter(z, x, y, zdir='z', c=color_list, s=0.6)


def visualize_df_voxel(tensor):
    """3D visualization for a distance field tensor, where each values on the boundary is represented through solid shapes.

    :param tensor: Distance Field (DF) tensor, np.array, shape = (32, 32, 32)
    :return: None
    """

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    volume = np.less_equal(tensor, 0.5)
    volumeTransposed = np.transpose(volume, (2, 0, 1))

    fig = plt.figure("Zero DF with voxels")
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(volumeTransposed)


def visualize_sdf_voxel(tensor):
    """3D visualization for a signed distance field tensor, where each values on the boundary is represented through solid shapes.

    :param tensor: Signed Distance Field (SDF) tensor, np.array, shape = (32, 32, 32)
    :return: None
    """

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    # find voxels with finite sdf values
    finite_region = np.isfinite(tensor) * 1

    # extract finite values
    volume = np.greater(finite_region, 0)
    volumeTransposed = np.transpose(volume, (2, 0, 1))

    fig = plt.figure("Finite SDF with voxels")
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(volumeTransposed)


def show():
    """ Shows all constructed figures. Mainly for visualizing multiple 3D figures at once
    """
    plt.show()


def generate_airplane_voxel_image(model, epoch, save_path, device):
    """ DF 3D tensor """

    input_file = "data/sample_airplane/sample_airplane.sdf"
    input_tensor = tensor_from_file(input_file)
    input_tensor[~np.isfinite(input_tensor).astype(np.bool)] = 0.0
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0)
    tensor = model(input_tensor.to(device)).detach().cpu().numpy()
    tensor = tensor.squeeze(0).squeeze(0)

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    volume = np.less_equal(tensor, 0.5)
    volumeTransposed = np.transpose(volume, (2, 0, 1))

    fig = plt.figure("Sample airplane predicted object")
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(volumeTransposed)

    # label figure
    plt.title("Sample airplane predicted object")

    save_path = os.path.join(save_path, '_sample_airplane_' + str(epoch) + '.png')
    plt.savefig(save_path)


def generate_voxel_image_from_model(dataset_type, model, tensor_path, idx, tensor_class, epoch, save_path, device):
    """ DF 3D tensor """

    input_file = tensor_path
    input_tensor = tensor_from_file(input_file)
    input_tensor[~np.isfinite(input_tensor).astype(np.bool)] = 0.0
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0)
    tensor = model(input_tensor.to(device)).detach().cpu().numpy()
    tensor = tensor.squeeze(0).squeeze(0)

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    volume = np.less_equal(tensor, 0.5)
    volumeTransposed = np.transpose(volume, (2, 0, 1))

    fig = plt.figure(dataset_type + ": predicted " + tensor_class + " " + str(idx))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(volumeTransposed)

    # label figure
    plt.title(dataset_type + ": predicted " + tensor_class + " " + str(idx))

    save_path = os.path.join(save_path, dataset_type + '_predicted_' + tensor_class + str(idx) + "_" + str(epoch) + '.png')
    plt.savefig(save_path)


def generate_original_voxel_image(dataset_type, tensor_path, idx, tensor_class, save_path):
    """ DF 3D tensor """

    target_tensor = tensor_from_file(tensor_path)
    tensor = torch.from_numpy(target_tensor).unsqueeze(0)

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    volume = np.less_equal(tensor, 0.5)
    volumeTransposed = np.transpose(volume, (2, 0, 1))

    fig = plt.figure(dataset_type + ": target " + tensor_class + " " + str(idx))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(volumeTransposed)

    # label figure
    plt.title(dataset_type + ": target " + tensor_class + " " + str(idx))

    save_path = os.path.join(save_path, dataset_type+ '_target_' + tensor_class + str(idx) + '.png')
    plt.savefig(save_path)


def plot_curves(loss, iou, accuracy, metadata):
    """ Plots the curves for a model run, given the arrays loss, iou, accuracy arrays and corresponsding metadata

    Args:
        loss: nxi array of losses, where n - number of curves to compare (usually train vs validation) and i -
              number of iterations for the plot

        iou: nxi array of intersection over union values, where n - number of curves to compare
             (usually train vs validation) and i - number of iterations for the plot

        accuracy: nxi array of accuracy, where n - number of curves to compare (usually train vs validation) and i -
                  number of iterations for the plot

    metadata: Array of n strings - what kind of data is present (usually "Train", "Validation")
    """
    plot_a_curve(loss, metadata, "Loss")
    plot_a_curve(iou, metadata, "IOU")
    plot_a_curve(accuracy, metadata, "Accuracy")


def plot_a_curve(data, metadata, curve_type):
    """ Plots the curves for a model run, given the arrays loss, iou, accuracy arrays and corresponding metadata.

    metadata: What kind of data is present (usually train, validation)
    curve_type: (Loss, IOU, Accuracy)
    """
    title = ""
    plt.title("Train vs Validation Error")
    n = len(data[0])  # number of epochs
    for i in range(len(metadata)):
        if i != 0:
            title += " vs "
        title += metadata[i]
        plt.plot(range(1, n + 1), data[i], label=metadata[i])
    title = title + " " + curve_type
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(curve_type)
    plt.legend(loc='best')
    plt.show()


def generate_curve(data, metadata, curve_type, save_path):
    """ Plots the curves for a model run, given the arrays loss, iou, accuracy arrays and corresponsding metadata

    metadata: What kind of data is present (usually train, validation)
    curve_type: (Loss, IOU, Accuracy)
    """
    title = ""

    # plot trajectories
    plt.figure()
    num_epochs = len(data[0])
    for i in range(len(metadata)):
        if i != 0:
            title += " vs "
        title += metadata[i]
        plt.plot(range(1, num_epochs + 1), data[i], label=metadata[i])
    title = title + " " + curve_type

    # label figure
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(curve_type)
    plt.legend(loc='best')

    save_path = os.path.join(save_path, curve_type + '_curve' + '.png')
    plt.savefig(save_path)


def compare_experiments(names, paths, metrics, save_path, epochs=15):
    """Generate plots comparing desired metrics across given models.
    args:
        :names: Names of the experiment corresponding to paths
        :paths: List of paths to the experiment directories
        :metrics: Desired metrics to generate plots of
        :save_path: path to save the plots
        :epochs: the number of data samples to plot
    """
    save_name = 'compare_{}epo_'.format(epochs)
    for name in names:
        save_name += name + '_'
    save_dir = os.path.join(save_path, save_name)
    os.mkdir(save_dir)

    # load data from experiment directories
    data = {}
    for name, path in zip(names, paths):
        if name not in data:
            data[name] = {}
        for metric in metrics:
            data[name][metric] = {}
            for split in ['train', 'val']:
                data[name][metric][split] = np.load(os.path.join(path, '{}_{}'.format(split, metric)))

    # generate and save plots
    for metric in metrics:
        plt.figure()
        plt.title("{} Model Comparison".format(metric))
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend(loc='best')
        for name in names:
            for split in ['train', 'val']:
                plt.plot(range(1, epochs+1), data[name][metric][split][:epochs], label='{}_{}'.format(name, split))

        plt.savefig(os.path.join(save_dir, '{}_curve.png'.format(metric)))


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = ['l2', 'iou', 'acc', 'loss']
    names = ['baseline', 'residual']
    paths = ['exp/baseline_15epochs/', 'exp/residual_unet']
    paths = [os.path.join(base_path, p, 'logs') for p in paths]
    compare_experiments(names, paths, metrics, os.path.join(base_path, 'exp'))
