import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


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
    """3D visualization for a distance field tensor, where each values on the boundary is 	       represented through solid shapes.

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
    """3D visualization for a signed distance field tensor, where each values on the boundary is 	       represented through solid shapes.

    :param tensor: Signed Distance Field (SDF) tensor, np.array, shape = (32, 32, 32)
    :return: None
    """

    if len(tensor.shape) == 4:
        tensor = np.squeeze(tensor, 0)

    # find voxels with finite sdf values
    finite_region = np.isfinite(tensor)*1

    # extract finite values
    volume = np.greater(finite_region, 0)
    volumeTransposed = np.transpose(volume, (2, 0, 1))
    
    fig = plt.figure("Finite SDF with voxels")
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(volumeTransposed)

def show():
    plt.show()
