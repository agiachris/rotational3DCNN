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

    # display full distance field
    x, y, z = np.indices(tensor.shape).reshape(3, -1)
    color = tensor[x, y, z]

    #fig = plt.figure("Full DF")
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x, -z, y, zdir='z', c=cm.jet(color), s=0.6)
    #plt.show()

    # The following portion attempts to extract the shape from the DF
    # # display zero voxels in DF
    x, y, z = np.where(tensor <= 0.5)
    color = tensor[x, y, z]
    #
    fig2 = plt.figure("Zero DF")
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(z, x, y, zdir='z', s=150)
    plt.show()
    
    # # display non-zeros voxels in DF
    x, y, z = tensor.nonzero()
    color = tensor[x, y, z]
    #
    #fig3 = plt.figure("Non-Zero DF")
    #ax3 = fig3.add_subplot(111, projection='3d')
    #ax3.scatter(z, x, y, zdir='z', c=cm.jet(color), s=0.6)
    #plt.show()


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
    fig = plt.figure("Finite Region")
    ax = fig.add_subplot(111, projection='3d')
    color_list = []
    for i in range(len(x)):
        color_list.append(m.to_rgba(finite_color[i]))
    ax.scatter(x, -z, y, zdir='z', c=color_list, s=0.6)

    # For visualization on voxels with infinite values
    # # find voxels with infinite sdf values
    # infinite_regions = ~np.isfinite(tensor)
    # infinite_color = tensor[infinite_regions]
    # z_inf, x_inf, y_inf = infinite_regions.nonzero()
    #
    # # plot infinite SDF voxels
    # fig2 = plt.figure("Infinite Region")
    # ax2 = fig2.add_subplot(111, projection='3d')
    # ax2.scatter(z_inf, x_inf, y_inf, zdir='z', c=cm.jet(infinite_color), s=0.6)
    # plt.show()
