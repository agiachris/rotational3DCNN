# rotational3DCNN

Note: this project was done exclusively by Christopher Agia, Polina Govorkova, YiFei Tang and Chao Glen Xu. The github account "nicknagi" corresponds to the account on the GPU computer and was not an additional contributor for this project.

### APS360 Final Project - Rotation Invariant 3D Shape Completion Network

This project aims to design a system capable of reconstructing 3-dimensional objects from a partially complete voxelized representation. Provided that the baseline task is completed with considerable results, we will investigate the effects of partial object rotation on the model’s performance, and attempt to design for a rotationally invariant reconstruction network.

### How to use

The training script expects two flags currently:
```python
python train.py --config config/path --gpuid gpuid_optional 
```

To evaluate a pre-trained model on a specific set:
```python
python evaluate.py --config config/path --model path/to/model --data data_split --gpuid gpuid_optional
```

### Completed Tasks
- [x] Configuration files (Chris)
- [x] Configurable training framework, trainer.py (Chris) 
- [x] ShapeNet dataset, dataset/shapenet.py (Chris)
- [x] Partition dataset into independent splits (Chris)
- [x] Baseline PyTorch model (YiFei)
- [x] 3D tensor metric computations (Polina)
- [x] 3D tensor visualization (Polina)
- [x] Training and evaluation loop (Glen)
- [x] Metric tracker and plotting (Chris & Polina)
- [x] Visualization class (Polina)
- [x] Evalutor class for model evaluation (YiFei & Chris)
- [x] Integrate visualization into training / validation (Polina)
- [x] Design variations of the baseline model (Team)
