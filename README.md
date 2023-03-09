# Introduction

This repository contains the Adverbs in Recipes (AIR) dataset and the code for 
the CVPR 23 paper: 

**Learning Action Changes by Measuring Verb-Adverb Textual Relationships**

### Cite

If you find our work useful, please cite our CVPR 23 paper:

```
@article{moltisanti23learning,
author = {Moltisanti, Davide and Keller, Frank and Bilen, Hakan and Sevilla-Lara, Laura},
title = {{Learning Action Changes by Measuring Verb-Adverb Textual Relationships}},
journal = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2023}}
```

### Authors

- [Davide Moltisanti](https://www.davidemoltisanti.com/research/)
- [Frank Keller](https://homepages.inf.ed.ac.uk/keller/)
- [Hakan Bilen](https://homepages.inf.ed.ac.uk/hbilen/)
- [Laura Sevilla-Lara](https://laurasevilla.me/)

All authors are at the University of Edinburgh.

# The AIR (Adverbs in Recipes) dataset

Adverbs in Recipes (AIR) is our new dataset specifically collected for adverb recognition. 
AIR is a subset of [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) where recipe videos show actions 
performed in ways that change according to an adverb (e.g. `chop thinly/coarsely`). 
AIR was carefully reviewed to ensure reliable annotations. Below are some properties of AIR:

| Property            | Value |
|---------------------|:-----:|
| Videos              |  7003 |
| Verbs               |   48  |
| Adverbs             |   10  |
| Pairs               |  186  |
| Avg. video duration |  8.4s |

AIR is organised in three CSV files, which you will find under the `./air` folder of this repository:

- `antonyms.csv`: lists the 10 adverbs and the respective antonyms
- `test_df.csv`: test split of the dataset
- `train_df.csv`: train split of the dataset

The train and test CSV files contain the following columns:

| Column name        | Explanation                                                   |
|--------------------|---------------------------------------------------------------|
| seg_id             | unique ID of the trimmed video segment                        |
| youtube_id         | YouTube ID of the untrimmed video                             |
| sentence           | parsed caption sentence aligned with the segment              |
| verb_pre_mapping   | the parsed verb before clustering                             |
| adverb_pre_mapping | the parsed adverb before clustering                           |
| clustered_verb     | the parsed verb after clustering                              |
| clustered_adverb   | the parsed adverb after clustering                            |
| start_time         | start time in seconds, relative to the untrimmed video        |
| end_time           | end time in seconds, relative to the untrimmed video          |
| duration           | duration of the segment in seconds                            |
| rgb_frames         | number of frames in the segment (extracted with original FPS) |
| verb_label         | numeric verb label                                            |
| adverb_label       | numeric adverb label                                          |                     

## AIR video samples

You can [watch our short video](https://youtu.be/YPNw35vtyu8) on YouTube to have a look at a few samples from AIR.

## S3D features

We provide pre-extracted S3D features [here](https://github.com/dmoltisanti/air-cvpr23/releases/tag/feat_stack%3D16_stride%3D1).

These were obtained with stack size equal to 16 frames and stride equal to 1 second (as specified in the paper).

Features are stored as PyTorch dictionaries with the following structure:

- `features`:
  - `${seg_id}`:
    - `s3d_features`: Tx1024 PyTorch tensor, where T is the number of stacks. These are the video features named `mixed_5c` in S3D
    - `video_embedding_joint_space`:  Tx512 PyTorch tensor, where T is the same as above. These are the video-text joint embeddings named `video_embedding` in S3D     
- `metadata`:
  - `${seg_id}`:
    - `start_time`: start time of the segment (in seconds)
    - `end_time`: end time of the segment (in seconds)
    - `clustered_verb`: verb associated with the segment
    - `clustered_adverb`: adverb associated with the segment
    - `frame_samples`: frame indices (relative to the trimmed video) sampled to extract the features

# Code

You can use the Python script `run.py` to try our method. Usage below:

``` 
usage: run.py [-h] [--train_batch TRAIN_BATCH] [--train_workers TRAIN_WORKERS] [--test_batch TEST_BATCH] [--test_workers TEST_WORKERS] [--lr LR] [--dropout DROPOUT] [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS] [--run_tags RUN_TAGS] [--tag TAG] [--test_frequency TEST_FREQUENCY] [--hidden_units HIDDEN_UNITS]
              [--s3d_init_folder S3D_INIT_FOLDER] [--no_antonyms] [--fixed_d] [--cls_variant]
              train_df_path test_df_path antonyms_df features_path output_path

Code to run the model presented in the CVPR 23 paper "Learning Action Changes by Measuring Verb-Adverb Textual Relationships"

positional arguments:
  train_df_path         Path to the datasets train set
  test_df_path          Path to the datasets test set
  antonyms_df           Path to the datasets antonyms csv
  features_path         Path to the pre-extracted S3D features
  output_path           Where you want to save logs, checkpoints and results

options:
  -h, --help            show this help message and exit
  --train_batch TRAIN_BATCH
                        Training batch size
  --train_workers TRAIN_WORKERS
                        Number of workers for the training data loader
  --test_batch TEST_BATCH
                        Testing batch size
  --test_workers TEST_WORKERS
                        Number of workers for the testing data loader
  --lr LR               Learning rate
  --dropout DROPOUT     Dropout for the model
  --weight_decay WEIGHT_DECAY
                        Weight decay for the optimiser
  --epochs EPOCHS       Number of training epochs
  --run_tags RUN_TAGS   What arguments should be used to create a run id, i.e. an output folder name
  --tag TAG             Any additional tag to be added to the run id
  --test_frequency TEST_FREQUENCY
                        How often the model should be evaluated on the test set
  --hidden_units HIDDEN_UNITS
                        Number of layers and hidden units for the models MLP, to be specified as a string of comma-separated integers. For example, 512,512,512 will create 3 hidden layers, each with 512 hidden units
  --s3d_init_folder S3D_INIT_FOLDER
                        Path to the S3D checkpoint. This will be downloaded automatically if you run setup.bash. Otherwise, you can download them from https://github.com/antoine77340/S3D_HowTo100M
  --no_antonyms         Whether you want to discard antonyms. See paper for more details
  --fixed_d             Runs the regression variant with fixed delta equal to 1. See paper for more details
  --cls_variant         Runs the classification variant. See paper for more details
```

We provide the following bash training scripts to run the variants of the method: 

- `scripts/train_cls.bash`: classification variant, trains the model with Cross Entropy
- `scripts/train_reg.bash`: regression variant, trains the model with MSE with delta calculated measuring verb-adverb textual relationships
- `scripts/train_reg_fixed_d.bash`: regression variant, trains the model with MSE with fixed `delta=1`
- `scripts/train_reg_no_ant.bash`: like `REG`, but discards antonyms.

See the paper for more details. 

### Setup

You can set up your environment with `bash setup.bash`. This will use conda to install the necessary dependencies, 
which you can find below and in the `environment.yaml` file:

```yaml
name: air_adverbs_cvpr23
channels:
  - defaults
dependencies:
  - pytorch::pytorch
  - pytorch::torchvision
  - nvidia::cudatoolkit=11
  - numpy
  - pip
  - scikit-learn
  - tqdm
  - pandas
  - tensorboard
  - pip:
    - opencv-python
    - yt-dlp
```

The setup script will also download [AIR's S3D features](https://github.com/dmoltisanti/air-cvpr23/releases/tag/feat_stack%3D16_stride%3D1) 
and [S3D's weights](https://github.com/antoine77340/S3D_HowTo100M).

### Output files

The code will save the following files at the specified output location:

```
├── logs
│   └── events.out.tfevents.1678189879.your_pc.1005191.0
├── model_output
│   ├── best_test_acc_a.pth
│   ├── best_test_map_m.pth
│   └── best_test_map_w.pth
├── model_state
│   ├── best_test_acc_a.pth
│   ├── best_test_map_m.pth
│   └── best_test_map_w.pth
├── summary.csv
```

Details about each file:

- `logs`: contains Tensorboard log files
- `model_output`: contains the output of the model (as a PyTorch dictionary)
  - one file for the best version of the model for each of the 3 main metrics
- `model_state`: contains the model parameters (as a PyTorch dictionary)
  - one file for the best version of the model for each of the 3 main metrics
- `summary_csv`: CSV file containing training and testing information for each epoch: 
  - `train_loss`: average training loss
  - `test_loss`: average test loss
  - `test_map_w`: mAP W, i.e. mean average precision with weighted averaging
  - `test_map_m`: mAP M, i.e. mean average precision with macro averaging
  - `test_acc_a`: Acc-A, i.e. binary accuracy of the GT adverbs versus its antonym
  - `test_no_act_gt_map_w`: mAP W calculated discarding the GT verb action label
  - `test_no_act_gt_map_m`: mAP M calculated discarding the GT verb action label
  - `test_no_act_gt_acc_a`: Acc-A calculated discarding the GT verb action label

### License

We release our code, annotations and features under the [MIT license](https://github.com/dmoltisanti/air-cvpr23/blob/main/LICENSE).

### Other repositories

Some small bits of code come from:

- [Action Modifiers](https://github.com/hazeld/action-modifiers)
- [S3D](https://github.com/antoine77340/S3D_HowTo100M)

These bits are clearly marked in the code.

# Pre-processing AIR 

You don't need to download videos, extract frames or [S3D features](https://github.com/dmoltisanti/air-cvpr23/releases/tag/feat_stack%3D16_stride%3D1)
to use our code.

Nevertheless, we provide our pre-processing scripts in this repository. You can learn how to use these scripts below.  

### Download videos

You can download and trim the videos from YouTube using the Python script `download_clips.py`. Usage below:

```
usage: download_clips.py [-h] [--n_proc N_PROC] [--use_youtube_dl] train_df_path test_df_path output_path yt_cookies_path

Download and trims videos from YouTube using either youtube-dlp or yt-dl (which you should install yourself). It uses 
multiprocessing to speed up downloading and trimming

positional arguments:
  train_df_path     Path to the train sets csv file
  test_df_path      Path to the test sets csv file
  output_path       Where you want to save the videos
  yt_cookies_path   Path to a txt file storing your YT cookies. This is needed to download some age-restricted videos. You can use https://github.com/hrdl-github/cookies-txt to store cookies in a text file
  
options:
  -h, --help        show this help message and exit
  --n_proc N_PROC   How many process you want to run in parallel to download videos
  --use_youtube_dl  Use youtube-dl instead of the default yt-dlp (not recommended as yt-dlp is now deprecated in favour of youtube-dl)
```

you can also refer to the bash script located at `scripts/download_air_clips.bash`.

### Extract RGB frames

To extract RGB frames from the video (e.g. if you want to then extract features yourself) you can use the Python 
script `extract_rgb_frames.py`. Usage below:

```
usage: extract_rgb_frames.py [-h] [--frame_height FRAME_HEIGHT] [--quality QUALITY] [--ext EXT] [--cuda] train_df_path test_df_path videos_path output_path

Extract RGB frames from videos using ffmpeg. If you have an ffmpeg compiled with cuda specify --cuda. You might need to deactivate the conda environment to call the cuda-enabled ffmpeg

positional arguments:
  train_df_path         Path to the train sets csv file
  test_df_path          Path to the test sets csv file
  videos_path           Path to the videos
  output_path           Where you want to save the frames

options:
  -h, --help            show this help message and exit
  --frame_height FRAME_HEIGHT
                        Height of the frame for resizing, in pixels
  --quality QUALITY     ffmpeg quality of the extracted image
  --ext EXT             Video file extension
  --cuda                Use ffmpeg with cuda hardware acceleration. If you have an ffmpeg compiled with cuda specify --cuda. You might need to deactivate the conda environment to call the cuda-enabled ffmpeg
```

you can also refer to the bash script located at `scripts/extract_air_rgb_frames.bash`.

### Extract S3D features

In case you want to extract features yourself you can use the Python script `extract_s3d_features.py`. Usage below:

``` 
usage: extract_s3d_features.py [-h] [--stack_size STACK_SIZE] [--stride STRIDE] [--batch_size BATCH_SIZE] [--workers WORKERS] [--jpg_digits JPG_DIGITS] [--jpg_prefix JPG_PREFIX] [--rgb_height RGB_HEIGHT] [--crop_size CROP_SIZE] train_df_path test_df_path frames_path output_path s3d_init_folder

Extract S3D features

positional arguments:
  train_df_path         Path to the train sets csv file
  test_df_path          Path to the test sets csv file
  frames_path           Path to the extracted frames
  output_path           Where you want to save the features
  s3d_init_folder       Path to the S3D checkpoint folder. You can find the model weights at https://github.com/antoine77340/S3D_HowTo100M

options:
  -h, --help            show this help message and exit
  --stack_size STACK_SIZE
                        Number of frames to be stacked as input to S3D
  --stride STRIDE       Stride in seconds
  --batch_size BATCH_SIZE
                        Batch size to extract features in parallel
  --workers WORKERS     Number of workers for the data loader
  --jpg_digits JPG_DIGITS
                        Number of digits in the JPG frames file name
  --jpg_prefix JPG_PREFIX
                        Prefix of the JPG frames file name
  --rgb_height RGB_HEIGHT
                        Height of the frame, in pixels
  --crop_size CROP_SIZE
                        Size of the central crop to be fed to S3D, in pixels
```

You will need to [download the S3D weights from the original repository](https://github.com/antoine77340/S3D_HowTo100M).

You can also refer to the bash script located at `scripts/extract_air_s3d_features.bash`, which will download S3D's 
weights automatically.

# Bash scripts

All bash scripts should be run from the repository's root, e.g. `bash ./scripts/train_reg.bash`.

# Help

If you need help don't hesitate to open a GitHub issue!

