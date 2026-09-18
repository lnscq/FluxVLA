"""Record the first fixed evaluation batch, not success-selected demos."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fluxvla.rl.experiment import require_wandb


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('Run demos in tmux')
    root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
    source = Path(__file__).resolve().parents[1]
    training = root / 'results/adjust_bottle_pi05_8gpu_resume1_20260917_120908'
    output = root / 'results' / args.tag
    output.mkdir(exist_ok=False)
    require_wandb()
    reports = {}
    for label, step in (('best', 30), ('last', 200)):
        name = f'{args.tag}_{label}_step{step}'
        experiment = root / 'results' / name
        if experiment.exists():
            raise FileExistsError(experiment)
        original = root / ('results/adjust_bottle_final200_20260918_1453_'
                           f'{label}_step{step}_test')
        expected = {}
        for rank in range(2):
            rows = (
                original /
                f'episodes/test/worker_{rank}.jsonl').read_text().splitlines()
            for line in rows[:5]:
                row = json.loads(line)
                expected[row['seed']] = row['success']
        command = [
            sys.executable, '-u', '-m', 'fluxvla.rl.eval',
            '--config-name=robotwin_adjust_bottle_eval_fluxvla_pi05_8gpu',
            f'runner.ckpt_path={training}/checkpoints/global_step_{step}',
            f'runner.logger.experiment_name={name}',
            'runner.logger.logger_backends=[wandb,tensorboard]',
            'env.eval.rollout_epoch=1', 'env.eval.video_cfg.save_video=false',
            f'+env.eval.full_video_dir={experiment}/full_videos'
        ]
        save(
            output / f'{label}_protocol.json', {
                'command':
                command,
                'started_at':
                time.time(),
                'expected_first_batch':
                expected,
                'selection': ('First five episodes per original worker; '
                              'no filtering by outcome.'),
                'scope': ('Illustrative replay, not new benchmark '
                          'or checkpoint selection.'),
            })
        print(
            json.dumps({
                'phase': label,
                'log': str(output / f'{label}.log')
            }),
            flush=True)
        with (output / f'{label}.log').open('x') as stream:
            subprocess.run(
                command,
                cwd=source,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True)
        actual = {}
        for path in (experiment / 'episodes/test').glob('worker_*.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['seed'] in actual:
                    raise RuntimeError('Duplicate replay seed')
                actual[row['seed']] = row['success']
        if actual != expected:
            save(output / f'{label}_replay_mismatch.json', {
                'expected': expected,
                'actual': actual
            })
            raise RuntimeError('Replay differs from original batch; '
                               'inspect recording before proceeding')
        recordings = []
        for path in sorted(
            (experiment /
             'full_videos').glob('worker_*/seed_*/recording.json')):
            record = json.loads(path.read_text())
            if not record['complete'] or record['success'] != expected[
                    record['seed']]:
                raise RuntimeError(
                    f'Incomplete/inconsistent recording: {path}')
            for view in ('head', 'three_cameras'):
                video = path.parent / f'{view}.mp4'
                probe = json.loads(
                    subprocess.check_output([
                        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries',
                        ('stream=width,height,r_frame_rate,nb_frames'
                         ':format=duration'), '-of', 'json',
                        str(video)
                    ],
                                            text=True))
                stream = probe['streams'][0]
                if int(
                        stream['nb_frames']
                ) != record['frames'] or stream['r_frame_rate'] != '25/1':
                    raise RuntimeError(f'Bad video frame count/FPS: {video}')
                record[view] = {'path': str(video), 'probe': probe}
            recordings.append(record)
        if len(recordings) != 10 or {row['seed']
                                     for row in recordings} != set(expected):
            raise RuntimeError('Missing or duplicate demo recordings')
        reports[label] = {
            'step': step,
            'replay_matches_original': True,
            'recordings': recordings
        }
        save(output / f'{label}_recordings.json', reports[label])
        print(
            json.dumps({
                'phase': f'{label}_complete',
                'episodes': len(recordings),
                'frames': [row['frames'] for row in recordings]
            }),
            flush=True)
    save(output / 'demo_index.json', reports)
    print(
        json.dumps({
            'phase': 'complete',
            'index': str(output / 'demo_index.json')
        }),
        flush=True)


if __name__ == '__main__':
    main()
