# Extract S3D features

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import torchvision.transforms.functional as F
import dataset
from run import to_cuda
from s3dg import S3D


def create_parser():
    parser = argparse.ArgumentParser(add_help=True, description='Extract S3D features')
    parser.add_argument('train_df_path', type=Path, help='Path to the train set''s csv file')
    parser.add_argument('test_df_path', type=Path, help='Path to the test set''s csv file')
    parser.add_argument('frames_path', type=Path, help='Path to the extracted frames')
    parser.add_argument('output_path', type=Path, help='Where you want to save the features')
    parser.add_argument('s3d_init_folder', type=Path, help='Path to the S3D checkpoint folder. You can find the model '
                                                           'weights at https://github.com/antoine77340/S3D_HowTo100M')
    parser.add_argument('--stack_size', default=16, type=int, help='Number of frames to be stacked as input to S3D')
    parser.add_argument('--stride', default=1, type=int, help='Stride in seconds')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size to extract features in parallel')
    parser.add_argument('--workers', default=8, type=int, help='Number of workers for the data loader')
    parser.add_argument('--jpg_digits', default=6, type=int, help='Number of digits in the JPG frames'' file name')
    parser.add_argument('--jpg_prefix', default='frame_', help='Prefix of the JPG frames'' file name')
    parser.add_argument('--rgb_height', default=226, type=int, help='Height of the frame, in pixels')
    parser.add_argument('--crop_size', default=224, type=int, help='Size of the central crop to be fed to S3D, '
                                                                   'in pixels')

    return parser


class VideoCenterCrop(torchvision.transforms.CenterCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, stack):
        return torch.stack([F.center_crop(s, self.size) for s in stack])


class DatasetsForFeaturesExtraction(Dataset):
    def __init__(self, df, frames_path, stride, stack_size, jpg_digits=6, jpg_prefix='frame_', rgb_height=226,
                 crop_size=224):
        self.frames_path = frames_path
        self.stride = stride
        self.stack_size= stack_size
        self.samples_per_seg = 1
        self.df_original = df
        self.df = self.expand_along_time()
        self.jpg_prefix = jpg_prefix
        self.jpg_digits = jpg_digits
        self.rgb_height = rgb_height
        self.transforms = transforms.Compose([VideoCenterCrop(crop_size)])

    def expand_along_time(self):
        rows = []

        for i, row in tqdm(self.df_original.iterrows(), desc='Expanding df', total=len(self.df_original),
                           file=sys.stdout):
            start_sec, end_sec = row['start_time'], row['end_time']
            duration_sec = end_sec - start_sec + 1
            max_n = row['rgb_frames']
            fps = np.round(max_n / duration_sec)
            frequency_fps = int(self.stride * fps)

            for s in np.arange(1, max_n, frequency_fps):
                row_t = dict(row)
                fe = min(s + frequency_fps - 1, max_n - 1)
                row_t['start_frame'] = s
                row_t['rgb_frames'] = fe
                rows.append(row_t)

        return pd.DataFrame(rows)

    def build_rgb_path(self, segment, frame_number):
        seg_id = dataset.Dataset.get_seg_id(segment)
        return self.frames_path / seg_id / f'{self.jpg_prefix}{str(frame_number).zfill(self.jpg_digits)}.jpg'

    def load_rgb_frames(self, segment, frame_samples):
        frames = []

        for frame_number in frame_samples:
            frame_path = self.build_rgb_path(segment, frame_number)
            img = cv2.imread(str(frame_path))  # IMPORTANT: cv images are BGR!
            assert img is not None, f'Could not load img {frame_path}'
            w, h, c = img.shape

            if w < self.rgb_height or h < self.rgb_height:
                d = float(self.rgb_height) - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

            img = (img / 255.)
            frames.append(img)

        frames = np.asarray(frames, dtype=np.float32)
        frames = torch.from_numpy(frames.transpose([3, 0, 1, 2]))

        return frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        segment = self.df.iloc[item]
        frame_samples = np.linspace(segment.start_frame, segment.rgb_frames, self.stack_size, dtype=int)
        frames = self.load_rgb_frames(segment, frame_samples)
        frames = self.transforms(frames)
        metadata = {k: getattr(segment, k) for k in ('seg_id', 'start_time', 'end_time', 'clustered_adverb',
                                                     'clustered_verb') if hasattr(segment, k)}
        metadata['frame_samples'] = frame_samples

        return frames, metadata


def setup_data(args):
    train_df = pd.read_csv(args.train_df_path)
    test_df = pd.read_csv(args.test_df_path)
    train_dataset = DatasetsForFeaturesExtraction(train_df, args.frames_path, args.stride, args.stack_size,  
                                                  jpg_digits=args.jpg_digits, jpg_prefix=args.jpg_prefix, 
                                                  rgb_height=args.rgb_height, crop_size=args.crop_size)
    test_dataset = DatasetsForFeaturesExtraction(test_df, args.frames_path, args.stride, args.stack_size,
                                                 jpg_digits=args.jpg_digits, jpg_prefix=args.jpg_prefix,
                                                 rgb_height=args.rgb_height, crop_size=args.crop_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.workers)

    return train_loader, test_loader


def setup_s3d(args):
    init_dict_path = args.s3d_init_folder / 's3d_dict.npy'
    s3d_checkpoint_path = args.s3d_init_folder / 's3d_howto100m.pth'
    print(f'Loading S3D weights from {args.s3d_init_folder}')
    original_classes = 512
    s3d = S3D(init_dict_path, num_classes=original_classes)
    s3d.load_state_dict(torch.load(s3d_checkpoint_path))

    if torch.cuda.is_available():
        s3d = s3d.cuda()
        s3d = torch.nn.DataParallel(s3d)

    return s3d


def collate_features(output, metadata):
    collated_output = {}
    collated_metadata = {}
    vids = np.array(metadata['seg_id'])
    u_ids = np.unique(vids)
    metadata_np = {k: np.array(v) for k, v in metadata.items()}

    for vid in u_ids:
        idx = vid == vids
        vf = {k: v[idx] for k, v in output.items()}

        collated_output[vid] = vf
        collated_metadata[vid] = {k: v[idx] for k, v in metadata_np.items() if k != 'seg_id'}

    return collated_output, collated_metadata


def extract_features(model, train_loader, test_loader, args):
    model.eval()

    with torch.no_grad():
        for set_, loader in zip(('train', 'test'), (train_loader, test_loader)):
            set_output = {}
            set_metadata = {}

            for frames, batch_metadata in tqdm(loader, desc=f'Extracting features for {set_} set', file=sys.stdout):
                frames = to_cuda((frames, ))[0]
                batch_output = model(frames)
                # renaming keys with better names
                batch_output['s3d_features'] = batch_output.pop('mixed_5c')
                batch_output['video_embedding_joint_space'] = batch_output.pop('video_embedding')

                for k, v in batch_output.items():
                    if k not in set_output:
                        set_output[k] = []

                    set_output[k].append(v.detach())

                for k, v in batch_metadata.items():
                    if k not in set_metadata:
                        set_metadata[k] = []

                    if isinstance(v, list):
                        set_metadata[k].extend(v)
                    elif isinstance(v, torch.Tensor):
                        set_metadata[k].extend(v.tolist())
                    else:
                        raise RuntimeError(f'Define how to combine metadata for type {type(v)}')

            set_output = {k: torch.cat([x.unsqueeze(0) if x.ndim == 1 else x for x in v], dim=0) if v else None
                          for k, v in set_output.items()}

            collated_output, collated_metadata = collate_features(set_output, set_metadata)
            f_path = args.output_path / f'stack={args.stack_size}_stride={args.stride}' / f'{set_}.pth'
            f_path.parent.mkdir(parents=True, exist_ok=True)
            f_dict = dict(features=collated_output, metadata=collated_metadata)
            torch.save(f_dict, f_path)
            print(f'Features for set {set_} saved to {f_path}')


def main():
    parser = create_parser()
    args = parser.parse_args()
    train_loader, test_loader = setup_data(args)
    s3d = setup_s3d(args)
    extract_features(s3d, train_loader, test_loader, args)


if __name__ == '__main__':
    main()
