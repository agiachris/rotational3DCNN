import os
import yaml
import torch
from models import UNet3dBaseline


def get_config(file_path):
    full_path = os.path.realpath(file_path)
    config = yaml.load(full_path)
    return config


def get_model(config):
    config['model_config'] = os
    if config['model_name'] == "baseline":
        return UNet3dBaseline.UNet3dBaseline(config['model_config'])