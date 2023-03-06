import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# if you have ffmpeg compiled with cuda
# remember to deactivate conda otherwise cuda-enabled system ffmpeg won't be called


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('train_df_path', type=Path)
    parser.add_argument('test_df_path', type=Path)
    parser.add_argument('videos_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--frame_height', default=256, type=int)
    parser.add_argument('--quality', default=4, type=int)
    parser.add_argument('--ext', default='mp4', type=str)

    return parser


def extract_frames_for_video(video_path, output_path, frame_height=256, quality=4):
    video_output_path = Path(output_path) / video_path.stem
    video_output_path.mkdir(exist_ok=True, parents=True)
    output_ = f'{video_output_path}/frame_%06d.jpg'

    # important: do not resample videos otherwise we will have issue with optical flow (lots of grey frames)
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-hwaccel', 'cuda', '-i', video_path,
                                '-vf', f'scale=-2:{frame_height}',
                                '-qscale:v', str(quality), '-y', output_],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # TODO add parameter to choose not to use cuda

    _, ffmpeg_err = process.communicate()
    ffmpeg_err = ffmpeg_err.decode()
    ffmpeg_err = ffmpeg_err.strip()
    ok = len(ffmpeg_err) == 0

    if not ok:
        tqdm.write(f'Error while extracting frames for {video_path}: {ffmpeg_err}')

    return ok, ffmpeg_err


def extract_frames_for_df(df, videos_path, output_path, ext='mp4', frame_height=256, quality=4):
    output_path = Path(output_path)
    video_clips = df['seg_id'].unique()
    videos_path = Path(videos_path)
    errors = []

    for v in tqdm(video_clips, desc='Extracting RGB frames', file=sys.stdout):
        video_path = videos_path / f'{v}.{ext}'

        if not video_path.exists():
            tqdm.write(f'Warning! Video not found: {video_path}')
            continue

        video_ok, err = extract_frames_for_video(video_path, output_path, frame_height=frame_height, quality=quality)

        if not video_ok:
            errors.append(dict(video_path=str(video_path), error=err))

    if errors:
        print('These clips could not be processed', errors)
        errors = pd.DataFrame(errors)
        errors.to_csv(output_path.parent / f'{output_path.stem}_errors.csv', index=False)
    else:
        print('No errors')

    print('All done')


def main():
    parser = create_parser()
    args = parser.parse_args()
    train_df = pd.read_csv(args.train_df_path)
    test_df = pd.read_csv(args.test_df_path)
    df = pd.concat([train_df, test_df])
    extract_frames_for_df(df, args.videos_path, args.output_path,
                          ext=args.ext, frame_height=args.frame_height, quality=args.quality)


if __name__ == '__main__':
    main()
