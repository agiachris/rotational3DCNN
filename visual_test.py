import os
from utils import *
from torch.utils.data import DataLoader
from dataset.visualization2 import *
import torch
from dataset.data_utils import *
from torch.utils.data import Dataset


if __name__ == "__main__":
	input_file = "/home/polinago/rotational3DCNN/data/1a04e3eab45ca15dd86060f189eb133__0__.sdf"
	target_file = "/home/polinago/rotational3DCNN/data/1a04e3eab45ca15dd86060f189eb133__0__.df"

	# get sdf input
	input_tensor = tensor_from_file(input_file)
	input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

	# get df target
	target_tensor = tensor_from_file(target_file)
	target_tensor = torch.from_numpy(target_tensor).unsqueeze(0)

	out = dict()
	out['inputs'] = input_tensor
	out['targets'] = target_tensor

	# visualization of a random sample
	sample = out
	#visualize_sdf(sample['inputs'].numpy())
	visualize_df(sample['targets'].numpy())

