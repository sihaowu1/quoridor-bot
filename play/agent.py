"""
Inference-side glue: load trained weights into the shared MCTS networks and
answer "what does the bot play here?".

Weight sources, in order of precedence:
  * AZ_CHECKPOINT=<path to training pickle>  -- the crash-safe checkpoint
    written by run.py (restores both networks via alphazero.checkpoint);
  * AZ_WEIGHTS_V / AZ_WEIGHTS_P              -- explicit .weights.h5 paths;
  * default: checkpoints/<GAME_NAME>_policy_{v,p}.weights.h5 in the repo.

Because mcts.predict_fn closes over the module-level policy_v / policy_p
instances, loading weights into them is all it takes for the search to play
with the trained network.
"""

import os

import numpy as np

from alphazero import mcts
from alphazero.game_config import GAME_NAME, GAME_OBS
from alphazero.mcts import Node, Policy_Player_MCTS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_networks():
    """Load trained weights into mcts.policy_v / mcts.policy_p.

    Returns a short description of what was loaded (for logging).
    """
    ckpt = os.environ.get('AZ_CHECKPOINT')
    if ckpt:
        from alphazero.checkpoint import load_checkpoint, restore_networks
        state = load_checkpoint(os.path.dirname(ckpt) or '.',
                                os.path.basename(ckpt))
        if state is None:
            raise FileNotFoundError(f'no readable checkpoint at {ckpt}')
        restore_networks(state, mcts.policy_v, mcts.policy_p, GAME_OBS)
        return f'checkpoint {ckpt}'

    default_dir = os.path.join(_ROOT, 'checkpoints')
    v_path = os.environ.get(
        'AZ_WEIGHTS_V',
        os.path.join(default_dir, f'{GAME_NAME}_policy_v.weights.h5'))
    p_path = os.environ.get(
        'AZ_WEIGHTS_P',
        os.path.join(default_dir, f'{GAME_NAME}_policy_p.weights.h5'))

    # Subclassed keras models must be built (called once) before
    # load_weights can assign into them.
    dummy = np.zeros((1, GAME_OBS), dtype=np.float32)
    mcts.policy_v(dummy)
    mcts.policy_p(dummy)
    mcts.policy_v.load_weights(v_path)
    mcts.policy_p.load_weights(p_path)
    return f'weights {v_path} + {p_path}'


def bot_move(game, observation):
    """Run a fresh MCTS from the current position and pick the best move.

    ``observation`` must be the canonical observation for the player to
    move (i.e. the value returned by the last reset()/step() call).

    Returns (action, value): the greedy action in the canonical frame of
    the player to move, and the search's mean root value from that same
    player's perspective.
    """
    root = Node(game.clone(), False, None, observation, None)
    _, action, _, _, _ = Policy_Player_MCTS(root, greedy=True)
    value = root.T / root.N if root.N else 0.0
    return int(action), float(value)
