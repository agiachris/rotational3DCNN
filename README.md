# Rotation Invariant 3D Shape Completion Network
The project is completed as a part of APS360: Introduction to Machine Learning course offered by the University of Toronto in Fall 2020. The project was done exclusively by Christopher Agia, Polina Govorkova, YiFei Tang and Chao Glen Xu. The github account "nicknagi" corresponds to the account on the GPU computer and was not an additional contributor for this project.

For the detailed description of the project and its results please see the [Project Report](https://github.com/agiachris/rotational3DCNN/blob/main/project_description_and_results/ProjectReport.pdf).

## Goal and motivation
Recent years have brought forth various technologies that depend heavily on sensing capabilities via software. Examples include autonomous driving, robotic manufacturing, and augmented reality. A commonality amongst these tasks is that they require some form of interaction with the 3D world (e.g. virtual or physical), but must operate on sensor information that is inherently an incomplete representation (e.g. 2D images, 2D depth maps, sparse 3D point clouds). 

In this project, we address a component of this challenge by designing a deep learning solution capable of reconstructing 3-dimensional objects from a partially complete voxelized representation.

In this work, we represent the partial shapes of objects in terms of a perspective signed distance field (SDF, i.e. input) and infer a perspective-invariant distance field of the completed shape (DF, i.e. target), as illustrated in Fig. 1. Through this formulation, we hypothesize that the CNNs must learn rotationally or perspective invariant representations of objects belonging to the set of eight class categories contained in our dataset. 

<img src="https://github.com/agiachris/rotational3DCNN/blob/main/project_description_and_results/proposal_overview.png" height="250" />

## Data
The source of our data is the ShapeNetCore subset of the ShapeNet database, which has been curated for our particular task by Dai et al. They go through an extensive process to produce signed distance field (SDF) and distance field (DF) input-target pairs, which altogether form the Stanford Graphics Shape Completion Benchmark.

## Models
We propose two CNN models that incorporate a unique set of 3D convolutional modules. The CNNs are modeled after an encoder/decoder architecture which produces outputs with equivalent dimensions to the input. 
#### Residual U-Net 
The model features an encoder composed of Residual Blocks, a decoder composed of DoubleConv Blocks, and skip connections that add the encoder’s voxel features to those of the decoder at various scale spaces. The skip connections help to maintain the fine-grained details of the shape’s input, and in combination with the Residual Blocks promotes fast and stable learning with improved gradient flow to the CNN encoder. Voxel-wise addition is preferred to concatenation for computational efficiency and to reduce the kernel sizes in the decoder. 
#### SE Residual U-Net
The model has an encoder built from Squeeze and Excite (SE) Residual Blocks [13], but is otherwise identical to the first model. The SE blocks enable the network to learn inter-channel dependencies that could exploit complementing voxel features at minimal cost.

<img src="https://github.com/agiachris/rotational3DCNN/blob/main/project_description_and_results/proposal_system.png" height="400" />

## Results
Performance on the test set:
| Model           | L2 Error     | IOU       | Accuracy   | Loss (MSE) |
| :---------------|-------------:| ---------:| ----------:|-----------:|
| Residual U-Net  |     5.50e-04 |   5.42e-1 |    9.79e-1 |    3.26e-01|
|SE-Residual U-Net|   5.41e-04   |   4.75e-1 |    9.79e-1 |    3.15e-01|


## How to run the project

The training script expects two flags currently:
```python
python train.py --config config/path --gpuid gpuid_optional 
```

To evaluate a pre-trained model on a specific set:
```python
python evaluate.py --config config/path --model path/to/model --data data_split --gpuid gpuid_optional
```
