"""
Integration tests for the AlphaZero MCTS on Quoridor.

These run with UNTRAINED networks on purpose: the tactical assertions only
hold if the search itself is correct — legal-move masking, true terminal
values entering the tree, and negamax sign flipping during backpropagation
(the same guarantees test_alphazero_ttt.py established on tic-tac-toe, now
exercised on the Quoridor action space with wall placements and jumps).

Run: python -m alphazero.test_alphazero_quoridor
(game_config must be on Quoridor, which is the default.)
"""

from copy import deepcopy
import sys

from alphazero import game_config
from alphazero.quoridor import NUM_PAWN_ACTIONS, wall_action
from alphazero.replay_buffer import ReplayBuffer

if game_config.GAME_NAME != 'Quoridor':  # pragma: no cover
    sys.exit('game_config is not on Quoridor; unset AZ_GAME and re-run.')

from alphazero.game_config import BOARD_SIZE, GAME_ACTIONS, make_game
from alphazero.mcts import Node
from alphazero.run import self_play_episode

SIMULATIONS = 400


def _make_position(p1, p2, to_play=1, hw=(), vw=(), walls=(3, 3)):
    # set_state exists with the same signature on both env backends, so
    # these tests run unchanged under AZ_BACKEND=py and AZ_BACKEND=cpp.
    game = make_game()
    game.set_state(p1, p2, to_play, hw, vw, walls[0], walls[1])
    return game


def _search(game, simulations=SIMULATIONS):
    root = Node(deepcopy(game), False, None, game._obs(), None)
    for _ in range(simulations):
        root.explore()
    return root


def test_finds_immediate_win():
    # X one step from the goal row; action 0 (N) wins on the spot.
    game = _make_position(p1=(1, 2), p2=(3, 2))
    root = _search(game)
    _, action, _, _, _ = root.next(greedy=True)
    assert action == 0, f'expected winning move 0 (N), got {action}'


def test_finds_immediate_win_as_second_player():
    # Same tactic for player -1 (canonical N = absolute S toward the
    # bottom row); O must sit one step from its goal on ANY board size.
    game = _make_position(p1=(1, 0), p2=(BOARD_SIZE - 2, 2), to_play=-1,
                          walls=(0, 0))
    root = _search(game)
    _, action, _, _, _ = root.next(greedy=True)
    assert action == 0, f'expected winning move 0 (N), got {action}'


def test_negamax_terminal_value():
    # The winning child must carry Q ~ +1 from the parent's perspective
    # (i.e. -T/N of the child), proving the true terminal outcome is
    # backed up with the right sign.
    game = _make_position(p1=(1, 2), p2=(3, 2))
    root = _search(game)
    win_child = root.child[0]
    q_for_parent = -(win_child.T / win_child.N)
    assert q_for_parent > 0.9, f'winning move Q was {q_for_parent:.3f}'
    assert win_child.done and win_child.terminal_value == -1.0


def test_blocks_opponent_win():
    # X (at (1,2)) wins next move unless O immediately walls off row 0.
    # O is too far from its own goal to race, so the ONLY non-losing moves
    # are the two horizontal walls covering column 2 of the top edge:
    # absolute H(0,1) / H(0,2), i.e. canonical H(3,1) / H(3,2) for O.
    game = _make_position(p1=(1, 2), p2=(2, 0), to_play=-1, walls=(0, 3))
    root = _search(game, simulations=1500)
    _, action, _, _, _ = root.next(greedy=True)
    blocking = {wall_action('H', BOARD_SIZE - 2, 1, BOARD_SIZE),
                wall_action('H', BOARD_SIZE - 2, 2, BOARD_SIZE)}
    assert action in blocking, \
        f'expected a blocking wall {sorted(blocking)}, got {action}'


def test_only_legal_actions_expanded():
    game = _make_position(p1=(3, 2), p2=(2, 2), hw=[(1, 2)], walls=(1, 1))
    root = _search(game, simulations=50)
    assert sorted(root.child) == game.legal_actions()


def test_policy_target_shape():
    game = make_game()
    game.reset()
    root = _search(game, simulations=50)
    _, _, _, probs, _ = root.next()
    assert probs.shape == (GAME_ACTIONS,)
    assert abs(probs.sum() - 1.0) < 1e-9
    # from the start position S (action 1) is off the board -> illegal
    assert probs[1] == 0.0, 'policy target puts mass on an illegal action'
    legal = set(game.legal_actions())
    illegal_mass = sum(probs[a] for a in range(GAME_ACTIONS) if a not in legal)
    assert illegal_mass == 0.0


def test_self_play_episode_targets():
    # The buffer must hold the longest possible game (max_moves plies),
    # otherwise the deque evicts early positions and len(buffer) != moves.
    game = make_game()
    buffer = ReplayBuffer(buffer_size=game.max_moves, batch_size=8)
    winner, moves = self_play_episode(buffer)
    assert winner in (1, -1, 0)
    assert 0 < moves <= game.max_moves
    assert len(buffer) == moves

    experiences = list(buffer.memory)
    for exp in experiences:
        assert exp.v in (-1.0, 0.0, 1.0)
        assert exp.p.shape == (GAME_ACTIONS,)
        assert abs(exp.p.sum() - 1.0) < 1e-9

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
    from tensorflow import keras

    # Seeds python/numpy AND TensorFlow: the network weights were the one
    # unseeded input, making these tactical assertions flaky (an unlucky
    # untrained net can bury a win-in-1 at 9x9 within the simulation
    # budget).  Fully seeded, the run is deterministic — and because the
    # C++ and Python envs are lockstep-identical, AZ_BACKEND=cpp and
    # AZ_BACKEND=py must produce identical search decisions here.
    keras.utils.set_random_seed(0)

    tests = [
        test_finds_immediate_win,
        test_finds_immediate_win_as_second_player,
        test_negamax_terminal_value,
        test_blocks_opponent_win,
        test_only_legal_actions_expanded,
        test_policy_target_shape,
        test_self_play_episode_targets,
    ]
    for test in tests:
        test()
        print(f'{test.__name__}: ok')
    print('all tests passed')
