#!/usr/bin/env python3
"""Record/check the GPU stage budget without a forced-termination watchdog."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl'))
    parser.add_argument(
        '--phase',
        choices=['preflight', 'baseline', 'train', 'final'],
        required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    path = args.root / 'gpu_budget.json'
    now = time.time()
    try:
        with path.open('x') as stream:
            json.dump(
                {
                    'started_at': now,
                    'training_deadline': now + 6 * 3600,
                    'final_deadline': now + 8 * 3600
                },
                stream,
                indent=2)
    except FileExistsError:
        pass
    budget = json.loads(path.read_text())
    deadline = budget['final_deadline' if args.phase ==
                      'final' else 'training_deadline']
    remaining = deadline - time.time()
    if remaining <= 0:
        raise SystemExit(
            'GPU phase budget exhausted; automatic extension forbidden')
    env = dict(os.environ, FLUX_RL_BUDGET_FILE=str(path))
    print(
        json.dumps({
            'phase': args.phase,
            'budget': budget,
            'remaining_seconds': remaining
        }),
        flush=True)
    # The runner checks its budget at checkpoint boundaries. Do not kill the
    # driver or workers asynchronously while an update/checkpoint is in flight.
    raise SystemExit(subprocess.run(command, env=env, check=False).returncode)


if __name__ == '__main__':
    main()
