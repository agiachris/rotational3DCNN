import os
import random


def partition_files(data_dir, config_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Partition dataset into training, validation, and test splits by the provided ratios.
    The samples in each data split are independent. This script creates three .txt files
    which contain the filenames that compose each data split.

    :param data_dir: relative path to root dataset directory
    :param config_dir: relative path to config dataset directory
    :param train_ratio: train data split ratio
    :param val_ratio: validation data split ratio
    :param test_ratio: test data split ratio
    :return: None
    """

    osj = os.path.join

    # directory structure
    inputs_path = osj(data_dir, 'shapenet_dim32_sdf')
    targets_path = osj(data_dir, 'shapenet_dim32_df')

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
    return


if __name__ == "__main__":
    data_root = "../data"
    config_root = "../config"
    partition_files(data_root, config_root)
