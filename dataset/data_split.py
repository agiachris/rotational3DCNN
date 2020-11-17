import os
import csv
import numpy as np
from dataset.data_utils import class_mapping


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
    model_names = {}
    for class_id in os.listdir(inputs_path):
        # input class_id files
        input_class_path = osj(inputs_path, class_id)
        input_files[class_id] = sorted([osj(input_class_path, sdf_file) for sdf_file in os.listdir(input_class_path)])

        # track number of unique shape models
        for sdf_path in input_files[class_id]:
            _, sdf_file = os.path.split(sdf_path)
            model_id = os.path.splitext(sdf_file)[0].split('__')[0]
            if class_id not in model_names:
                model_names[class_id] = set()
            model_names[class_id].add(model_id)

        # target class_id files
        target_class_path = osj(targets_path, class_id)
        target_files[class_id] = sorted([osj(target_class_path, df_file) for df_file in os.listdir(target_class_path)])

    # determine input .sdf and target .df sample pairs
    stats = {}
    samples = {'train': [], 'valid': [], 'test': []}
    for class_id in input_files:
        stats[class_id] = dict()
        # randomly partition class specific shape models into train, validation, test splits
        n = len(model_names[class_id])
        models_np = np.array(list(model_names[class_id]))
        train_models = np.random.choice(models_np, size=int(n*train_ratio), replace=False)
        models_np = np.array([m for m in models_np if m not in train_models])
        valid_models = np.random.choice(models_np, size=int(n*val_ratio), replace=False)
        test_models = np.array([m for m in models_np if m not in valid_models])
        assert (n == len(train_models) + len(valid_models) + len(test_models))

        sample_count = train_samples = valid_samples = test_samples = 0
        for sdf_path in input_files[class_id]:
            # get model_id of input .sdf file
            _, sdf_file = os.path.split(sdf_path)
            model_id = os.path.splitext(sdf_file)[0].split('__')[0]
            sdf = osj(class_id, sdf_file)

            # ensure .df target file exists for desired model_id
            df = osj(class_id, model_id + '__0__.df')

            if os.path.exists(osj(targets_path, df)):
                sample_count += 1

            if model_id in train_models:
                train_samples += 1
                samples['train'].append((sdf, df))

            elif model_id in valid_models:
                valid_samples += 1
                samples['valid'].append((sdf, df))

            elif model_id in test_models:
                test_samples += 1
                samples['test'].append((sdf, df))

        # ensure no targets are missing
        assert (sample_count == len(input_files[class_id]))

        # store class-specific stats per data split
        stats[class_id]['train'] = (len(train_models), train_samples)
        stats[class_id]['valid'] = (len(valid_models), valid_samples)
        stats[class_id]['test'] = (len(test_models), test_samples)

    print("Data Partition Statistics")
    print("Class       |       Train (# models, # samples) | Valid (# models, # samples) | Test (# models, # samples)")
    for class_id in stats:
        class_stats = stats[class_id]
        print(class_mapping[class_id], '({})'.format(class_id),
              '|', class_stats['train'],
              '|', class_stats['valid'],
              '|', class_stats['test'])

    # write to independent csv files
    for data_split in samples:
        csv_file = osj(config_dir, data_split + '.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(samples[data_split])

    return


if __name__ == "__main__":
    root_dir = os.path.split(os.path.realpath(__file__))[0]
    root_dir = os.path.split(root_dir)[0]
    data_root = os.path.join(root_dir, "data")
    config_root = os.path.join(root_dir, "config")
    partition_files(data_root, config_root)
