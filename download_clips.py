import argparse
import subprocess
import sys
from functools import partial
from multiprocessing import Pool

import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('train_df_path', type=Path)
    parser.add_argument('test_df_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('yt_cookies_path', type=Path, help='Path to a txt file storing your YT cookies. This is '
                                                           'needed to download some age-restricted videos. '
                                                           'You can use https://github.com/hrdl-github/cookies-txt '
                                                           'to store cookies in a text file')
    parser.add_argument('--n_proc', default=12, type=int)
    parser.add_argument('--use_youtube_dl', action='store_true', help='Use youtube-dl instead of the default '
                                                                      'yt-dlp (not recommended)')
    return parser


def download_video_from_youtube(output_path, video_tuple, yt_cookies_path='', use_youtube_dl=False):
    _, video_tuple = video_tuple
    video_start = video_tuple.start_time
    video_end = video_tuple.end_time
    youtube_id = video_tuple.youtube_id
    output_path_video = output_path / Path(f'{youtube_id}_{int(video_start)}-{int(video_end)}.mp4')

    if output_path_video.exists() and is_video_ok(output_path_video):
        return True, video_tuple

    youtube_url = f'https://youtu.be/{youtube_id}'

    if use_youtube_dl:
        cmd = 'youtube-dl'
        f = 'best'
    else:
        cmd = 'yt-dlp'
        f = 'b'

    process = subprocess.Popen([cmd, '-f', f, '-g', '--cookies', yt_cookies_path, youtube_url],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vid_url, yt_err = process.communicate()
    vid_url = vid_url.strip()
    yt_err = yt_err.strip()

    if yt_err:
        print(f'Youtube error downloading {youtube_url}: {yt_err}')
        video_tuple['yt_err'] = yt_err
        return False, video_tuple

    # placing -ss and -to before -i is fast but not precise. Place them after -i if precision is important (but note
    # that this will first download all the video and then will trim it

    start = time.strftime('%H:%M:%S', time.gmtime(video_start))
    end = time.strftime('%H:%M:%S', time.gmtime(video_end))

    process = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-ss', start, '-to', end,
                                '-i', vid_url, '-c', 'copy', output_path_video],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    ffmpeg_out, ffmpeg_err = process.communicate()
    ffmpeg_err = ffmpeg_err.strip()

    if ffmpeg_err and not is_video_ok(output_path_video):
        video_tuple['ffmpeg_err'] = ffmpeg_err
        print(f'FFMpeg Error downloading {youtube_url}: {ffmpeg_err}')
        return False, video_tuple

    return True, video_tuple


def download(df, output_path, yt_cookies_path, n_proc=12, use_youtube_dl=False):
    df = df.drop_duplicates(subset='seg_id')
    func = partial(download_video_from_youtube, output_path, yt_cookies_path=yt_cookies_path,
                   use_youtube_dl=use_youtube_dl)
    errors = []
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    with Pool(processes=n_proc) as pool:
        for ok, t in tqdm(pool.imap_unordered(func, df.iterrows()), desc='Downloading clips', total=len(df),
                          file=sys.stdout):
            if not ok:
                errors.append(t)

    errors_df = pd.DataFrame(errors)
    errors_path = output_path.parent / f'{output_path.stem}_errors.csv'
    errors_path.parent.mkdir(exist_ok=True, parents=True)
    errors_df.to_csv(errors_path, index=False)


def is_video_ok(video_path, print_=False):
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'null', '-'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    ffmpeg_out, ffmpeg_err = process.communicate()
    ffmpeg_err = ffmpeg_err.strip()

    if ffmpeg_err:
        if print_:
            print(f'Caught this error while reading {video_path}: {ffmpeg_err}')

        return False
    else:
        return True


def main():
    parser = create_parser()
    args = parser.parse_args()
    train_df = pd.read_csv(args.train_df_path)
    test_df = pd.read_csv(args.test_df_path)
    df = pd.concat([train_df, test_df])
    download(df, args.output_path, args.yt_cookies_path, n_proc=args.n_proc, use_youtube_dl=args.use_youtube_dl)


if __name__ == '__main__':
    main()
