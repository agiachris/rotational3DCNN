import os
import numpy
from dataset.visualization import *
from dataset.data_utils import (tensor_from_file, class_mapping)
import matplotlib.pyplot as plt

if __name__ == '__main__':

    class_idx = 0

    class_list = list(class_mapping.keys())
    base_path = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]

    class_id = class_list[class_idx]
    print("Generating images of {}".format(class_mapping[class_id]))
    for i in range(5):
        # extract random file
        class_name = os.path.join('../data/shapenet_dim32_df', class_id)
        class_dir = os.path.join(base_path, class_name)
        filenames = os.listdir(class_dir)
        idx = int(np.random.random() * len(filenames))
        file = os.path.join(class_dir, filenames[idx])

        # plot distance field
        plt.figure()
        visualize_df_voxel(tensor_from_file(file))
        plt.show()

