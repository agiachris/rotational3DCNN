import os
import csv
import torch
from dataset.data_utils import *
from torch.utils.data import Dataset


class ShapeNet(Dataset):

    def __init__(self, config, config_path, split):
        """
        Load ShapeNet data of scanned partial models and the corresponding distance
        transforms of the complete models.
        """

        osj = os.path.join

        # directory structure
        data_path = config['data_path']
        inputs_path = osj(data_path, 'shapenet_dim32_sdf')
        targets_path = osj(data_path, 'shapenet_dim32_df')

        samples = list()
        csv_file = osj(config_path, split + '.csv')
        with open(csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                sdf_file = osj(inputs_path, row[0])
                df_file = osj(targets_path, row[1])
                samples.append((sdf_file, df_file))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_file, target_file = self.samples[idx]

        # get sdf input
        input_tensor = tensor_from_file(input_file)
        input_tensor[~np.isfinite(input_tensor).astype(np.bool)] = 0.0
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

        # get df target
        target_tensor = tensor_from_file(target_file)
        target_tensor[~np.isfinite(target_tensor).astype(np.bool)] = 0.0
        target_tensor = torch.from_numpy(target_tensor).unsqueeze(0)

        out = dict()
        out['inputs'] = input_tensor
        out['targets'] = target_tensor
        return out
