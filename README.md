# Quoridor AlphaZero Bot

## Repository Structure

```text
.
├── alphazero/
│   ├── game_config.py           # the game the pipeline trains on
│   ├── mcts.py                  # negamax PUCT search + shared networks
│   ├── nn.py                    # policy / value MLPs
│   ├── run.py                   # self-play training driver (auto-resumes)
│   ├── checkpoint.py            # crash-safe save/restore of training state
│   ├── quoridor.py              # Quoridor environment (pure-Python reference) + self-tests
│   ├── quoridor_cpp.py          # drop-in env wrapper around the C++ engine
│   ├── tic_tac_toe.py           # validation game + self-tests
│   ├── test_alphazero_quoridor.py
│   ├── test_alphazero_ttt.py
│   ├── test_cpp_backend.py      # lockstep C++-vs-Python cross-validation
│   └── replay_buffer.py
├── quoridor/                    # C++ engine
│   ├── quoridor.cpp / .h        # the engine (mirrors alphazero/quoridor.py exactly)
│   ├── bindings.cpp             # pybind11 module (alphazero.quoridor_engine)
│   ├── build_ext.py             # builds the extension with the bundled zig toolchain
│   └── main.cpp                 # standalone human-vs-human CLI
├── README.md
├── .python-version
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## Setup

Python 3.13 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## C++ engine backend

The Quoridor environment exists twice with identical semantics: the
pure-Python reference (`alphazero/quoridor.py`) and a C++ port
(`quoridor/quoridor.cpp`) exposed to Python via pybind11. Training uses the
C++ engine automatically once it is built; everything else is unchanged.

```bash
# build the extension module (alphazero/quoridor_engine.*.so); needs no
# system compiler — the ziglang dev dependency ships a clang toolchain
uv run python quoridor/build_ext.py

# cross-validate the two backends in lockstep (obs, rewards, legal actions,
# rendering, exceptions must be bit-identical at every ply of hundreds of
# random and rule-targeted games); --bench prints a speed comparison
uv run python -m alphazero.test_cpp_backend --bench
```

| env variable | default | meaning                                              |
|--------------|---------|------------------------------------------------------|
| `AZ_BACKEND` | `auto`  | `auto` (C++ if built), `cpp` (require), `py` (force) |

The two engines must be changed together — the lockstep suite is the
contract. `quoridor/main.cpp` additionally builds into a standalone
human-vs-human CLI on the same engine.

## Running

All commands run from the repository root.

```bash
# Quoridor environment rule tests (movement, jumps, walls, path checks,
# canonical symmetry, random playthroughs)
python -m alphazero.quoridor

# MCTS integration tests on Quoridor (untrained nets must still find
# forced wins and blocking walls if the search is correct)
python -m alphazero.test_alphazero_quoridor

# train the Quoridor bot by self-play; writes checkpoints, weights and
# plots (game_length.png, win_rate.png, *_loss.png) to checkpoints/ and
# automatically RESUMES from the latest checkpoint when re-run
python -m alphazero.run

# tic-tac-toe regression suites (the pipeline defaults to Quoridor;
# AZ_GAME=ttt points game_config back at tic-tac-toe)
python -m alphazero.tic_tac_toe
AZ_GAME=ttt python -m alphazero.test_alphazero_ttt
AZ_GAME=ttt python -m alphazero.run
```

## Checkpointing and crash recovery

`run.py` saves the complete training state — network weights, Adam
optimizer state, replay buffer, metric histories, RNG states — atomically
after every episode (`alphazero/checkpoint.py`), keeping the previous save
as a `.bak` fallback. Re-running `python -m alphazero.run` automatically
resumes from the latest readable checkpoint; a checkpoint from a different
game or board size is refused with a clear error.

| env variable          | default        | meaning                                 |
|-----------------------|----------------|-----------------------------------------|
| `AZ_CHECKPOINT_DIR`   | `checkpoints/` | where checkpoints, weights and plots go |
| `AZ_CHECKPOINT_EVERY` | `1`            | save frequency in episodes              |
| `AZ_EPISODES`         | `300`          | total episodes for the run              |
| `AZ_GAME`             | `quoridor`     | `quoridor` or `ttt`                     |

## Training on Google Colab

Open `alphazero/notebook.ipynb` in Colab and `Run all` on a GPU runtime —
it is the complete workflow: mounts Google Drive, clones/updates this repo,
builds the C++ engine with Colab's g++ (verifying the pipeline selected it),
optionally cross-validates the backends, points `AZ_CHECKPOINT_DIR` at
Drive and trains. After any crash or disconnect, `Run all` again: training
resumes exactly where it left off (same episode, optimizer momentum and
replay buffer) from the Drive checkpoint, and the plots cell can be re-run
mid-training to monitor progress.

Colab's preinstalled `tensorflow`, `numpy` and `matplotlib` plus a
`pip install pybind11` (done by the notebook) are all the pipeline needs —
no `uv sync` there.
