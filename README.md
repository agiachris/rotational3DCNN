# rotational3DCNN

### APS360 Final Project - Rotation Invariant 3D Shape Completion Network

This project aims to design a system capable of reconstructing 3-dimensional objects from a partially complete voxelized representation. Provided that the baseline task is completed with considerable results, we will investigate the effects of partial object rotation on the model’s performance, and attempt to design for a rotationally invariant reconstruction network.

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

### Open Tasks 
- [ ] Evaluator class for model evaluation (YiFei)
    - Create dataset flag in argparse
- [ ] Integrate visualization into training / validation (Polina)
    - Create parameters in yaml files for number of samples to visualize
- [ ] Deploy starter code on GPU server (Glen)
    - nvidia-smi --> gpuid flag, attempt training
- [ ] Design variations of the baseline model (Team)

Target deadline: Nov 23 am (Monday)
### Schedule
- Nov 23 (Monday, evening)
- Nov 25 (Wednesday, usual)
- Nov 27 (Friday, evening)
