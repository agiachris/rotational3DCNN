import os
from utils import *
from dataset.shapenet import ShapeNet
from torch.utils.data import DataLoader
from models.UNet3dBaseline import UNet3dBaseline
from dataset.visualization import *

if __name__ == "__main__":
    # Carry out various tests related to data loading
    path = '/home/agiachris/Documents/year_4/aps360/rotational3DCNN'
    file = 'config/main.yaml'

    config = get_config(path, file)
    dataset = ShapeNet(config)

    # visualization of a random sample
    sample = dataset.__getitem__(500)
    visualize_sdf(sample['inputs'].numpy())
    visualize_df(sample['targets'].numpy())

    # testing dataloader and simple forward pass
    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=True, num_workers=1, drop_last=False)
    model = UNet3dBaseline()

    for batch, data in enumerate(loader, 0):
        if batch == 1:
            break

        inputs = data['inputs']
        targets = data['targets']
        preds = model(inputs)

        assert(preds.size() == inputs.size())

