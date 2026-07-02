"""
Tic-tac-toe environment for validating the AlphaZero implementation.

The API mirrors gymnasium (``reset() -> (obs, info)``,
``step(action) -> (obs, reward, terminated, truncated, info)``) so the game
drops into the existing MCTS/training code, which only relies on that
interface plus ``deepcopy``.

Two-player conventions (these are what the MCTS must be adapted to, see the
fixes reported for mcts.py / run.py):

* Players are +1 (X, moves first) and -1 (O).
* The observation is always CANONICAL: a length-9 float vector from the
  perspective of the player about to move (+1 = own stones, -1 = opponent
  stones, 0 = empty).  One network can therefore evaluate both players.
* ``step`` returns the reward from the perspective of the player who JUST
  moved: +1 for completing a winning line, 0 for a draw or a non-terminal
  move, -1 for playing an illegal (occupied) square, which immediately loses.
  The illegal-move-loses rule lets an unmodified full-action-space MCTS run;
  masking with ``info["action_mask"]`` is the proper fix.
* ``info`` always carries ``to_play``, ``legal_actions``, ``action_mask`` and,
  when the game is over, ``winner`` (+1 / -1 / 0 for a draw).

Board layout / action indices::

    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

Run ``python tic_tac_toe.py`` to execute the self-tests, including an
exhaustive enumeration of the full game tree checked against the known
outcome counts (255168 games: 131184 X wins, 77904 O wins, 46080 draws).
"""

import numpy as np

GAME_ACTIONS = 9  # one action per square
GAME_OBS = 9      # canonical board vector

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)

_SYMBOLS = {1: 'X', -1: 'O', 0: '.'}


class TicTacToe:

    def __init__(self):
        self.reset()

    def reset(self, seed=None, options=None):
        # The board is a plain Python list (not a numpy array) so that
        # clone()/step() stay cheap inside deep tree searches.
        self.board = [0] * 9
        self.to_play = 1
        self.done = False
        self.winner = None  # +1 / -1 / 0 (draw), None while running
        return self._obs(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError('step() called on a finished game; call reset()')
        if not 0 <= action < GAME_ACTIONS:
            raise ValueError(f'action must be in [0, {GAME_ACTIONS}), got {action}')

        mover = self.to_play

        if self.board[action] != 0:
            # Illegal move: immediate loss for the mover.
            self.done = True
            self.winner = -mover
            return self._obs(), -1.0, True, False, self._info()

        self.board[action] = mover

        if self._is_win(mover):
            self.done = True
            self.winner = mover
            reward = 1.0
        elif all(self.board):
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        self.to_play = -mover
        return self._obs(), reward, self.done, False, self._info()

    def legal_actions(self):
        return [a for a in range(GAME_ACTIONS) if self.board[a] == 0]

    def clone(self):
        """Cheap copy for tree search (also backs __deepcopy__)."""
        c = TicTacToe.__new__(TicTacToe)
        c.board = self.board.copy()
        c.to_play = self.to_play
        c.done = self.done
        c.winner = self.winner
        return c

    def __deepcopy__(self, memo):
        # The existing MCTS copies environments with deepcopy; route it
        # through clone() so expansion doesn't pay generic-deepcopy costs.
        return self.clone()

    def render(self):
        print(str(self))

    def __str__(self):
        rows = (' '.join(_SYMBOLS[self.board[r * 3 + c]] for c in range(3))
                for r in range(3))
        return '\n'.join(rows)

    def _obs(self):
        return np.array([c * self.to_play for c in self.board], dtype=np.float64)

    def _info(self):
        info = {
            'to_play': self.to_play,
            'legal_actions': self.legal_actions(),
            'action_mask': np.array([c == 0 for c in self.board], dtype=np.int8),
        }
        if self.done:
            info['winner'] = self.winner
        return info

    def _is_win(self, player):
        return any(all(self.board[i] == player for i in line) for line in WIN_LINES)


# ---------------------------------------------------------------------------
# Self-tests: python tic_tac_toe.py
# ---------------------------------------------------------------------------

def _play(moves):
    """Play a move sequence from the start; return (env, last step tuple)."""
    env = TicTacToe()
    result = None
    for move in moves:
        result = env.step(move)
    return env, result


def _test_win_detection():
    # Every line, for both players, must be detected the moment it completes.
    for player in (1, -1):
        for line in WIN_LINES:
            env = TicTacToe()
            env.board = [0] * 9
            env.board[line[0]] = env.board[line[1]] = player
            # Give the opponent two stones elsewhere so the position is
            # plausible; exact placement is irrelevant to the check.
            others = [i for i in range(9) if i not in line][:2]
            for i in others:
                env.board[i] = -player
            env.to_play = player

            obs, reward, terminated, truncated, info = env.step(line[2])
            assert terminated and reward == 1.0 and info['winner'] == player, \
                f'missed win on line {line} for player {player}'
            assert not truncated

    # A full line of the OPPONENT's stones must not count as a win.
    env = TicTacToe()
    env.board = [-1, -1, -1, 1, 1, 0, 0, 0, 0]
    env.to_play = 1
    assert not env._is_win(1)
    assert env._is_win(-1)


def _test_draw():
    # X: 0, 8, 6, 1, 5 / O: 4, 2, 7, 3 -> no line for either side.
    env, (obs, reward, terminated, truncated, info) = _play(
        [0, 4, 8, 2, 6, 7, 1, 3, 5])
    assert terminated and reward == 0.0 and info['winner'] == 0
    assert env.legal_actions() == []


def _test_illegal_move():
    env = TicTacToe()
    env.step(4)                      # X takes the centre
    obs, reward, terminated, _, info = env.step(4)  # O plays the same square
    assert terminated and reward == -1.0 and info['winner'] == 1

    # Stepping a finished game must raise.
    try:
        env.step(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError('step() on a finished game did not raise')

    # Out-of-range actions must raise.
    try:
        TicTacToe().step(9)
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range action did not raise')


def _test_canonical_observation():
    env = TicTacToe()
    obs, info = env.reset()
    assert obs.shape == (GAME_OBS,) and not obs.any()
    assert info['to_play'] == 1 and len(info['legal_actions']) == 9

    obs, _, _, _, info = env.step(0)
    # It is now O's turn: X's stone must appear as -1 (opponent) to O.
    assert info['to_play'] == -1 and obs[0] == -1.0

    obs, _, _, _, info = env.step(4)
    # Back to X: own stone +1 at 0, O's stone -1 at 4.
    assert info['to_play'] == 1 and obs[0] == 1.0 and obs[4] == -1.0
    assert info['action_mask'].sum() == 7
    assert sorted(info['legal_actions']) == [1, 2, 3, 5, 6, 7, 8]


def _test_clone_independence():
    env = TicTacToe()
    env.step(0)
    from copy import deepcopy
    for copy_fn in (TicTacToe.clone, deepcopy):
        c = copy_fn(env)
        c.step(4)
        assert env.board[4] == 0 and c.board[4] == -1
        assert env.to_play == -1 and c.to_play == 1


def _test_exhaustive_enumeration():
    """Enumerate every legal game; totals must match the known counts."""
    counts = {1: 0, -1: 0, 0: 0}

    def recurse(env):
        for action in env.legal_actions():
            child = env.clone()
            _, reward, terminated, _, info = child.step(action)
            if terminated:
                assert info['winner'] != -info['to_play'] or reward == 1.0
                counts[info['winner']] += 1
            else:
                recurse(child)

    recurse(TicTacToe())
    assert counts == {1: 131184, -1: 77904, 0: 46080}, counts
    assert sum(counts.values()) == 255168


def _test_random_playthroughs():
    import random
    rng = random.Random(0)
    for _ in range(500):
        env = TicTacToe()
        moves = 0
        terminated = False
        while not terminated:
            _, reward, terminated, truncated, info = env.step(
                rng.choice(env.legal_actions()))
            moves += 1
            assert not truncated and reward in (0.0, 1.0)
        assert moves <= 9 and info['winner'] in (1, -1, 0)


if __name__ == '__main__':
    tests = [
        _test_win_detection,
        _test_draw,
        _test_illegal_move,
        _test_canonical_observation,
        _test_clone_independence,
        _test_random_playthroughs,
        _test_exhaustive_enumeration,
    ]
    for test in tests:
        test()
        print(f'{test.__name__}: ok')
    print('all tests passed')
