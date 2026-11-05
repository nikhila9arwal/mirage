#!/usr/bin/env python3
"""Replay FIFO behavior and test accounting; optionally compile GPU replays.

This does not allocate a GPU or launch the model. CXX and NVCC select compilers.
Generated binaries stay outside Git in research/nas/.build.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

SESSION = Path(__file__).resolve().parents[1]
ROOT = SESSION.parents[1]
OPTIONS = {'baseline': (0, 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-gpu', action='store_true')
    args = parser.parse_args()
    output = SESSION / '.build'
    output.mkdir(exist_ok=True)
    common = ['-I' + str(ROOT / 'include/mirage/persistent_kernel'),
              '-DNAS_SELECTOR_FIFO=1', '-DNAS_SELECTOR_HISTOGRAM=0',
              '-DNAS_MOE_LAYERS=48', '-DNAS_MOE_STATE_SLOTS=16']
    source = str(SESSION / 'tests/test_fifo.cc')
    for p, r in ((1, 1), (4, 8)):
        for name, (cache, skip) in OPTIONS.items():
            flags = [f'-DNAS_PREFILL_CHUNK_ROWS={p}', f'-DNAS_MOE_TOP_K={r}',
                     f'-DNAS_FIFO_CACHE_OLDEST={cache}',
                     f'-DNAS_FIFO_SKIP_STATIC_CHECKS={skip}']
            binary = output / f'fifo_p{p}_r{r}_{name}'
            subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O2',
                            '-Wall', '-Wextra', '-pedantic', *common, *flags,
                            source, '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
            if args.compile_gpu and (p, r) == (4, 8):
                subprocess.run([os.environ.get('NVCC', 'nvcc'), '-std=c++17',
                                '-O2', '-arch=sm_90a', '-x', 'cu', *common,
                                *flags, source, '-o',
                                str(output / f'fifo_gpu_{name}')], check=True)
    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'test_analysis',
                    'test_dispatch_analysis', 'test_selector_analysis'],
                   cwd=SESSION / 'measurement', check=True)


if __name__ == '__main__':
    main()
