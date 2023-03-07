#!/bin/bash

python run.py \
./air/train_df.csv \
./air/test_df.csv \
./air/antonyms.csv \
./air/s3d_features/stack=16_stride=1 \
./experiments/air