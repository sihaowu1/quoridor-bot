"""
Build the C++ Quoridor engine as a CPython extension module.

    uv run python quoridor/build_ext.py

No system compiler is required: the `ziglang` dev dependency ships a full
clang/lld toolchain, so this works on a bare machine (and inside WSL) as
long as `uv sync` has run.  The compiled module lands at

    alphazero/quoridor_engine.<abi-tag>.so

and imports as ``alphazero.quoridor_engine`` (wrapped by
alphazero/quoridor_cpp.py; selected via AZ_BACKEND in game_config.py).
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / 'quoridor' / 'quoridor.cpp', ROOT / 'quoridor' / 'bindings.cpp']
OUTPUT = ROOT / 'alphazero' / ('quoridor_engine' + sysconfig.get_config_var('EXT_SUFFIX'))


def zig_command():
    """Prefer the venv's ziglang package; fall back to a zig/c++ on PATH."""
    try:
        import ziglang
        zig = Path(ziglang.__path__[0]) / ('zig.exe' if os.name == 'nt' else 'zig')
        if zig.exists():
            return [str(zig), 'c++']
    except ImportError:
        pass
    from shutil import which
    for compiler in (['zig', 'c++'], ['c++'], ['g++'], ['clang++']):
        if which(compiler[0]):
            return compiler
    raise SystemExit('no C++ compiler found: run `uv sync` to install ziglang')


def main():
    cmd = zig_command() + [
        '-std=c++17', '-O3', '-DNDEBUG', '-fPIC', '-shared',
        '-fvisibility=hidden',
        f'-I{pybind11.get_include()}',
        f'-I{sysconfig.get_paths()["include"]}',
        *map(str, SOURCES),
        '-o', str(OUTPUT),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(f'built {OUTPUT.relative_to(ROOT)}')

    # Smoke test in a fresh interpreter so a broken build fails here, not
    # at training time.
    check = ('import alphazero.quoridor_engine as q; e = q.Engine(5, 3); '
             'e.step(0); assert len(e.legal_actions()) > 0; '
             'assert e.observation().shape == (q.obs_size(5),)')
    subprocess.run([sys.executable, '-c', check], check=True, cwd=ROOT)
    print('import smoke test passed')


if __name__ == '__main__':
    main()
