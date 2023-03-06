#!/bin/bash

echo "Creating conda environment"
conda env create -f environment.yaml

echo "Downloading S3D weights. See https://github.com/antoine77340/S3D_HowTo100M"
wget -P ./s3d_init_folder https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget -P ./s3d_init_folder https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy

# TODO download features once repo is public