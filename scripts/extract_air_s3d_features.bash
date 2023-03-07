#!/bin/bash

echo "Downloading S3D weights. See https://github.com/antoine77340/S3D_HowTo100M"
wget -P ./s3d_init_folder -N "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth"
wget -P ./s3d_init_folder -N "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy"

python extract_s3d_features.py \
./air/train_df.csv \
./air/test_df.csv \
./air/frames/rgb \
"./air/s3d_features" \
./s3d_init_folder \
--workers 12 \
--stride 1 \
--stack_size 16