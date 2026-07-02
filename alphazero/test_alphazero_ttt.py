"""
Integration tests for the AlphaZero MCTS on tic-tac-toe.

These run with UNTRAINED networks on purpose: the tactical assertions only
hold if the search itself is correct — legal-move masking, true terminal
values entering the tree, and negamax sign flipping during backpropagation.
A search missing any of those fixes fails these tests regardless of how the
networks are initialised.

Run: AZ_GAME=ttt python -m alphazero.test_alphazero_ttt
(game_config defaults to Quoridor; the env var points it back here.)
"""

from copy import deepcopy
import random
import sys

import numpy as np

from alphazero import game_config

if game_config.GAME_NAME != 'TicTacToe':  # pragma: no cover
    sys.exit('game_config is not on TicTacToe; '
             'run with AZ_GAME=ttt python -m alphazero.test_alphazero_ttt')

from alphazero.tic_tac_toe import TicTacToe, GAME_ACTIONS
from alphazero.mcts import Node
from alphazero.replay_buffer import ReplayBuffer
from alphazero.run import self_play_episode

SIMULATIONS = 400


def _make_position(board, to_play):
    game = TicTacToe()
    game.board = list(board)
    game.to_play = to_play
    return game


def _search(game, simulations=SIMULATIONS):
    root = Node(deepcopy(game), False, None,
                np.array([c * game.to_play for c in game.board],
                         dtype=np.float64),
                None)
    for _ in range(simulations):
        root.explore()
    return root


def test_finds_immediate_win():
    # X: 0, 4  O: 1, 3  -> X to move; 8 completes the 0-4-8 diagonal.
    game = _make_position([1, -1, 0,
                           -1, 1, 0,
                           0, 0, 0], to_play=1)
    root = _search(game)
    _, action, _, _, _ = root.next(greedy=True)
    assert action == 8, f'expected winning move 8, got {action}'


def test_blocks_opponent_win():
    # X: 0, 1  O: 4  -> O to move; X threatens 2 (top row), O must block.
    game = _make_position([1, 1, 0,
                           0, -1, 0,
                           0, 0, 0], to_play=-1)
    root = _search(game)
    _, action, _, _, _ = root.next(greedy=True)
    assert action == 2, f'expected blocking move 2, got {action}'


def test_negamax_terminal_value():
    # From test_finds_immediate_win's position, the winning child must carry
    # Q ~ +1 from the parent's perspective (i.e. -T/N of the child), proving
    # the true terminal outcome is backed up with the right sign.
    game = _make_position([1, -1, 0,
                           -1, 1, 0,
                           0, 0, 0], to_play=1)
    root = _search(game)
    win_child = root.child[8]
    q_for_parent = -(win_child.T / win_child.N)
    assert q_for_parent > 0.9, f'winning move Q was {q_for_parent:.3f}'
    assert win_child.done and win_child.terminal_value == -1.0


def test_only_legal_actions_expanded():
    game = _make_position([1, -1, 0,
                           -1, 1, 0,
                           0, 0, 0], to_play=1)
    root = _search(game, simulations=50)
    assert sorted(root.child) == [2, 5, 6, 7, 8]


def test_policy_target_shape():
    game = TicTacToe()
    game.step(4)  # occupy the centre so one action is illegal
    root = _search(game, simulations=50)
    _, _, _, probs, _ = root.next()
    assert probs.shape == (GAME_ACTIONS,)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[4] == 0.0, 'policy target puts mass on an illegal action'


def test_self_play_episode_targets():
    buffer = ReplayBuffer(buffer_size=100, batch_size=8)
    winner, moves = self_play_episode(buffer)

    assert winner in (1, -1, 0)
    assert 5 <= moves <= 9  # fastest possible win is 5 plies
    assert len(buffer) == moves

    experiences = list(buffer.memory)
    for i, exp in enumerate(experiences):
        assert exp.v in (-1.0, 0.0, 1.0)
        assert exp.p.shape == (GAME_ACTIONS,)
        assert abs(exp.p.sum() - 1.0) < 1e-9
        # position i has 9 - i empty squares; the policy target may only
        # put mass on that many actions
        assert (exp.p > 0).sum() <= 9 - i

    # value targets are outcomes from the mover's perspective, so along one
    # decisive game they alternate sign; in a draw they are all zero
    values = [exp.v for exp in experiences]
    if winner == 0:
        assert all(v == 0.0 for v in values)
    else:
        assert values[-1] == 1.0  # the player who made the last move won
        for i in range(len(values) - 1):
            assert values[i] == -values[i + 1]


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    tests = [
        test_finds_immediate_win,
        test_blocks_opponent_win,
        test_negamax_terminal_value,
        test_only_legal_actions_expanded,
        test_policy_target_shape,
        test_self_play_episode_targets,
    ]
    for test in tests:
        test()
        print(f'{test.__name__}: ok')
    print('all tests passed')
