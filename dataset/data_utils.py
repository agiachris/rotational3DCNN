import numpy as np
import os

class_mapping = {'03001627': 'chair',
                 '04379243': 'table',
                 '03636649': 'lamp',
                 '02958343': 'car',
                 '02933112': 'cabinet',
                 '02691156': 'airplane',
                 '04256520': 'couch',
                 '04530566': 'boat'}

def tensor_from_file(filename):
    """Load Signed Distance Field (SDF) inputs and Distance Field (DF) targets
    from custom format provided by http://graphics.stanford.edu/projects/cnncomplete/
    """
    # load data
    data = np.fromfile(filename, dtype=np.float32)
    # extract header, encoded in uint64
    header = data[:6]
    header = header.tobytes()
    header = np.frombuffer(header, dtype=np.uint64)
    # reshape data into dimensions provided in header
    return data[6:].reshape(header)

def extract_categories(val_set, number):
    class_samples = {'03001627': [],
                 '04379243': [],
                 '03636649': [],
                 '02958343': [],
                 '02933112': [],
                 '02691156': [],
                 '04256520': [],
                 '04530566': []}
    for item in val_set:
        folders, filename = os.path.split(item[0])
        class_folder = os.path.split(folders)[1]
        if len(class_samples[class_folder]) < number:
            if "__0__" in filename:
                class_samples[class_folder].append(item)
    return class_samples