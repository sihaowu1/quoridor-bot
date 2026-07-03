"""
Drop-in Quoridor environment backed by the C++ engine.

``QuoridorCpp`` exposes the same API, conventions and observable state as
the pure-Python reference implementation ``alphazero.quoridor.Quoridor``
(canonical observations/actions, mover-perspective rewards, gymnasium-style
``reset``/``step``, ``legal_actions``, ``clone``, ``pos``/``h_walls``/
``v_walls``/``walls_left`` accessors) — the two are kept bit-identical by
the lockstep tests in alphazero/test_cpp_backend.py.

Importing this module requires the compiled extension:

    uv run python quoridor/build_ext.py

game_config.py picks this backend automatically when the extension is
available (override with AZ_BACKEND=py / AZ_BACKEND=cpp).
"""

import os
import warnings

import numpy as np

from alphazero import quoridor_engine as _engine
from alphazero.quoridor import DEFAULT_BOARD_SIZE, DEFAULT_WALLS


def _check_engine_fresh():
    """Refuse a compiled engine older than the C++ sources.

    A stale quoridor_engine.so silently reintroduces whatever the sources
    have since fixed (training-impacting: e.g. an outdated max_moves
    default truncating games early).  Raising ImportError makes
    AZ_BACKEND=cpp fail loudly and AZ_BACKEND=auto fall back to the
    (correct, slower) Python env after the warning below.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = [os.path.join(root, 'quoridor', name)
               for name in ('quoridor.cpp', 'quoridor.h', 'bindings.cpp')]
    newest = max((os.path.getmtime(s) for s in sources if os.path.exists(s)),
                 default=0.0)
    if os.path.getmtime(_engine.__file__) < newest:
        msg = ('compiled Quoridor engine is older than the C++ sources; '
               'rebuild it with: uv run python quoridor/build_ext.py')
        warnings.warn(msg)
        raise ImportError(msg)


_check_engine_fresh()


class QuoridorCpp:

    def __init__(self, board_size=DEFAULT_BOARD_SIZE, walls=DEFAULT_WALLS,
                 max_moves=None):
        # C++ raises ValueError for board_size < 3, same as the Python env.
        self._e = _engine.Engine(board_size, walls, max_moves or 0)
        self.n = board_size
        self.max_walls = walls
        self.max_moves = self._e.max_moves
        self.num_actions = self._e.num_actions
        self.obs_dim = self._e.obs_dim

    # ------------------------------------------------------------------
    # public API used by the MCTS / training driver
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self._e.reset()
        return self._obs(), self._info()

    def step(self, action):
        reward, terminated, truncated = self._e.step(action)
        return self._obs(), reward, terminated, truncated, self._info()

    def legal_actions(self):
        return self._e.legal_actions()

    def clone(self):
        c = QuoridorCpp.__new__(QuoridorCpp)
        c._e = self._e.clone()
        c.n = self.n
        c.max_walls = self.max_walls
        c.max_moves = self.max_moves
        c.num_actions = self.num_actions
        c.obs_dim = self.obs_dim
        return c

    def __deepcopy__(self, memo):
        return self.clone()

    def render(self):
        print(str(self))

    def __str__(self):
        return str(self._e)

    # ------------------------------------------------------------------
    # observable state, mirroring the Python env's attributes
    # ------------------------------------------------------------------

    @property
    def to_play(self):
        return self._e.to_play

    @property
    def done(self):
        return self._e.done

    @property
    def winner(self):
        return self._e.winner if self._e.done else None

    @property
    def move_count(self):
        return self._e.move_count

    @property
    def pos(self):
        return {1: self._e.pawn(1), -1: self._e.pawn(-1)}

    @property
    def walls_left(self):
        return {1: self._e.walls_left(1), -1: self._e.walls_left(-1)}

    @property
    def h_walls(self):
        return set(self._e.h_walls())

    @property
    def v_walls(self):
        return set(self._e.v_walls())

    def set_state(self, p1, p2, to_play, h_walls, v_walls,
                  walls_p1, walls_p2):
        """Install an arbitrary position (testing / analysis)."""
        self._e.set_state(tuple(p1), tuple(p2), to_play,
                          [tuple(w) for w in h_walls],
                          [tuple(w) for w in v_walls],
                          walls_p1, walls_p2)

    # ------------------------------------------------------------------

    def _obs(self):
        return self._e.observation()

    def _info(self):
        legal = self._e.legal_actions()
        mask = np.zeros(self.num_actions, dtype=np.int8)
        mask[legal] = 1
        info = {
            'to_play': self._e.to_play,
            'legal_actions': legal,
            'action_mask': mask,
        }
        if self._e.done:
            info['winner'] = self._e.winner
        return info
