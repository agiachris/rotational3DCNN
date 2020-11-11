import os
import torch
from dataset.data_utils import *
from torch.utils.data import Dataset


class ShapeNet(Dataset):

    def __init__(self, config):
        """
        Load ShapeNet data of scanned partial models and the corresponding distance
        transforms of the complete models.
        """

        osj = os.path.join

        # directory structure
        data_path = config['data_path']
        inputs_path = osj(data_path, 'shapenet_dim32_sdf')
        targets_path = osj(data_path, 'shapenet_dim32_df')

        # group files according the class_id
        input_files = {}
        target_files = {}
        for class_id in os.listdir(inputs_path):
            # input class_id files
            input_class_path = osj(inputs_path, class_id)
            input_files[class_id] = [osj(input_class_path, sdf_file) for sdf_file in os.listdir(input_class_path)]
            # target class_id files
            target_class_path = osj(targets_path, class_id)
            target_files[class_id] = [osj(target_class_path, df_file) for df_file in os.listdir(target_class_path)]

        # determine input .sdf and target .df sample pairs
        sample_count = 0
        samples = []
        for class_id in input_files:
            for sdf_path in input_files[class_id]:
                # get model_id of input .sdf file
                _, sdf_file = os.path.split(sdf_path)
                model_id = os.path.splitext(sdf_file)[0].split('__')[0]

                # ensure .df target file exists for desired model_id
                df_file = osj(targets_path, class_id, model_id + '__0__.df')
                if df_file in target_files[class_id]:
                    samples.append((sdf_path, df_file))

                sample_count += 1

        assert (len(samples) == sample_count)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_file, target_file = self.samples[idx]

        # get sdf input
        input_tensor = tensor_from_file(input_file)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

        # get df target
        target_tensor = tensor_from_file(target_file)
        target_tensor = torch.from_numpy(target_tensor).unsqueeze(0)

        out = dict()
        out['inputs'] = input_tensor
        out['targets'] = target_tensor
        return out
