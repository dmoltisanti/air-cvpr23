#!/bin/bash

echo "Creating conda environment"
conda env create -f environment.yaml

echo "Downloading S3D weights. See https://github.com/antoine77340/S3D_HowTo100M"
wget -P ./s3d_init_folder "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth"
wget -P ./s3d_init_folder "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy"

wget -P ./air/s3d_features/stack=16_stride=1 "https://github.com/dmoltisanti/air-cvpr23/releases/download/feat_stack%3D16_stride%3D1/train.pth"
wget -P ./air/s3d_features/stack=16_stride=1 "https://github.com/dmoltisanti/air-cvpr23/releases/download/feat_stack%3D16_stride%3D1/test.pth"
