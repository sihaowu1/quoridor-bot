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
│   ├── replay_buffer.py
│   └── notebook.ipynb           # Google Colab notebook
├── quoridor/                    # C++ engine
│   ├── bindings.cpp             # expose C++ engine to pythin via pybind11
│   ├── build_ext.py             # builds engine as a CPython extension module
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

## Get started

The training runs on a Google Colab. Get started by installing the VSCode extension for Colab. 

Under "5 - Configure the Run", change the env variables if needed:

| env variable          | default        | meaning                                 |
|-----------------------|----------------|-----------------------------------------|
| `AZ_CHECKPOINT_DIR`   | `checkpoints/` | where checkpoints, weights and plots go |
| `AZ_CHECKPOINT_EVERY` | `1`            | save frequency in episodes              |
| `AZ_EPISODES`         | `300`          | total episodes for the run              |
| `AZ_GAME`             | `quoridor`     | `quoridor` or `ttt`                     |

Select Python 3 kernel and any GPU in alphazero/notebook.ipynb. 
Then, run all. 

Note: Colab is preinstalled with all necessary libraries, so don't ```uv sync```.

## Local Development

Python 3.13 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

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

## AlphaZero Algorithm

This section provides an explanation of the AlphaZero algorithm in simple terms. 

At the core, AlphaZero is two neural networks (nn) and a Monte Carlo Tree Search (MCTS). 

### MCTS

AlphaZero uses Monte Carlo Tree Search (MCTS). MCTS is a "smarter" brute-force algorithm, where it only checks promising game states rather than all game states. MCTS keeps track of a decision tree of game states and updates each node in these four steps: 
1. Selection: picking the most promising branch
2. Expansion: after step 1, pick the branch that has not been previously explored
3. Simulation: completes a game from choices in step 1 and 2
4. Backpropagation: updates parent node values with new information after it finished playing a game 

The tree stores statistics for moves, such as: 
* frequency of picking this move
* how good the move seems
* how good did the neural network think this move was

But, how does MCTS know what is the "most promising" branch? 

### Neural Networks

AlphaZero uses two neural networks (or combined into one): policy and value. Policy gives a probability distribution over all legal moves, where a higher probability represents a preference. Value gives a score for the current board state, where a more positive number means good and a more negative position means bad. 

### Training

AlphaZero trains via self-play. Each move, it does the following:

* Check the next possible moves from the current board state. Choose the move with either a low visit count or that looks promising according to the two nns. 
* Repeat until a game is done. 
* There's an outcome, so update the nns and the MCTS accordingly. 

### Visualizing the Board

The AlphaZero algorithm doesn't see the board the same way we do. 
It keeps track of the board as a flat 292-number vector (81 own-pall cells, 81 opponent-pawn cells, 64 horizontal-wall, etc.) 
So, AlphaZero has to relearn the rules for every board location. 

To make this more efficient, we arrange the board as a conv net, basically a stack of planes. 
This way, the machine can understand that walls block movement with one set of weights, 
instead of relearning it for every move. 