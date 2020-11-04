import h5py
import numpy as np
from torch.utils.data import Dataset


class ShapeNet(Dataset):

    def __init__(self, config):
        """
        Load hdf5 ShapeNet of scanned partial models and the corresponding distance
        transforms of the complete models.
        """

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return 0
