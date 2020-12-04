#!/bin/bash
eval "$(conda shell.bash hook)" &&\
conda activate aps360 &&\
python -c "print('Hello World')" &&\
# python train.py --config config/residual.yaml --gpuid 0
