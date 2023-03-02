#!/bin/bash

conda create --name adverb-grounding
conda activate adverb-grounding

# make sure to install the right cudatoolkit version (nvidia-smi shows the version)
conda install pytorch torchvision cudatoolkit=11 -c pytorch
conda install jupyterlab matplotlib pandas tqdm tabulate