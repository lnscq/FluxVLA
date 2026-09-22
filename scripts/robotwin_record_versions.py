#!/usr/bin/env python3
"""Record reproducibility inputs without credentials or environment dumps."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def output(*command):
    return subprocess.check_output(command, text=True).strip()


def main():
    root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
    source = Path(__file__).resolve().parents[1]
    files = list((source / 'fluxvla/rl').rglob('*.py'))
    files += list((source / 'configs/rl').rglob('*.yaml'))
    files += list((source / 'configs/rl/model').glob('*.py'))
    files += list((source / 'scripts').glob('*robotwin*'))
    files += [source / 'fluxvla/engines/utils/overwatch.py']
    sources = {
        str(path.relative_to(source)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files)) if path.is_file()
    }
    record = {
        'recorded_at':
        datetime.now(timezone.utc).isoformat(),
        'python':
        sys.version,
        'executable':
        sys.executable,
        'platform':
        platform.platform(),
        'commits': {
            'fluxvla':
            output('git', '-C', str(source), 'rev-parse', 'HEAD'),
            'rlinf':
            output('git', '-C',
                   os.environ.get('RLINF_ROOT', str(source.parent / 'RLinf')),
                   'rev-parse', 'HEAD'),
            'robotwin':
            output('git', '-C', str(root / 'src/RoboTwin'), 'rev-parse',
                   'HEAD'),
        },
        'integration_sha256':
        sources,
        'gpu':
        output('nvidia-smi',
               '--query-gpu=index,uuid,name,memory.total,driver_version',
               '--format=csv'),
        'gpu_process_snapshot':
        output('nvidia-smi',
               '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
               '--format=csv'),
        'budget':
        json.loads((root / 'gpu_budget.json').read_text()),
        'oidn':
        '2.3.3 (process-local preload, Blackwell support)',
        'gpu_exclusivity':
        'Inspect the process snapshot; exclusivity is not assumed.',
    }
    for label, interpreter in (
        ('flux-robotwin', root / 'envs/flux-robotwin/bin/python'),
        ('openpi-reference', root / 'envs/openpi-reference/bin/python'),
    ):
        (root / 'logs' / f'{label}-freeze.txt'
         ).write_text(output(str(interpreter), '-m', 'pip', 'freeze') + '\n')
    path = root / 'preflight/runtime_manifest.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
