"""
Lockstep cross-validation of the C++ Quoridor engine against the pure-Python
reference implementation.

Every test drives ``alphazero.quoridor.Quoridor`` (the correctness
reference) and ``alphazero.quoridor_cpp.QuoridorCpp`` (the pybind11-wrapped
C++ engine) through identical action sequences and asserts that EVERYTHING
observable is bit-identical at every ply: observations, rewards,
termination/truncation flags, info dicts, legal-action lists, action masks,
absolute pawn/wall/walls-left state, move counters, board rendering, and
exception behaviour.

Requires the compiled extension (uv run python quoridor/build_ext.py).

Run: python -m alphazero.test_cpp_backend
"""

import random

import numpy as np

from alphazero.quoridor import NUM_PAWN_ACTIONS, Quoridor, _make, wall_action
from alphazero.quoridor_cpp import QuoridorCpp


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------

def _assert_same_view(py, cpp, ctx=''):
    """Full observable-state comparison between the two backends."""
    assert py.to_play == cpp.to_play, ctx
    assert py.done == cpp.done, ctx
    assert py.winner == cpp.winner, ctx
    assert py.move_count == cpp.move_count, ctx
    assert py.pos == cpp.pos, ctx
    assert py.walls_left == cpp.walls_left, ctx
    assert py.h_walls == cpp.h_walls, ctx
    assert py.v_walls == cpp.v_walls, ctx
    assert list(py.legal_actions()) == list(cpp.legal_actions()), ctx
    assert str(py) == str(cpp), ctx
    if not py.done:
        assert np.array_equal(py._obs(), cpp._obs()), ctx


def _assert_same_info(info_p, info_c, ctx=''):
    assert info_p['to_play'] == info_c['to_play'], ctx
    assert list(info_p['legal_actions']) == list(info_c['legal_actions']), ctx
    assert np.array_equal(info_p['action_mask'], info_c['action_mask']), ctx
    assert ('winner' in info_p) == ('winner' in info_c), ctx
    if 'winner' in info_p:
        assert info_p['winner'] == info_c['winner'], ctx


def _step_both(py, cpp, action, ctx=''):
    obs_p, r_p, term_p, trunc_p, info_p = py.step(action)
    obs_c, r_c, term_c, trunc_c, info_c = cpp.step(action)
    ctx = f'{ctx} action={action}'
    assert (r_p, term_p, trunc_p) == (r_c, term_c, trunc_c), ctx
    assert obs_c.dtype == obs_p.dtype == np.float64, ctx
    assert np.array_equal(obs_p, obs_c), ctx
    _assert_same_info(info_p, info_c, ctx)
    _assert_same_view(py, cpp, ctx)
    return r_p, term_p, trunc_p, info_p


def _pair(p1, p2, to_play=1, hw=(), vw=(), walls=(3, 3), n=5, max_moves=None):
    """The same arbitrary position installed in both backends."""
    py = _make(p1, p2, to_play, hw, vw, walls, n, max_moves)
    cpp = QuoridorCpp(n, 3, max_moves)
    cpp.set_state(p1, p2, to_play, hw, vw, walls[0], walls[1])
    _assert_same_view(py, cpp, f'setup p1={p1} p2={p2}')
    return py, cpp


def _lockstep_game(rng, n, walls, pawn_bias=0.0, ctx=''):
    """Play one full random game in lockstep; returns the winner."""
    py = Quoridor(n, walls)
    cpp = QuoridorCpp(n, walls)
    obs_p, info_p = py.reset()
    obs_c, info_c = cpp.reset()
    assert np.array_equal(obs_p, obs_c), ctx
    _assert_same_info(info_p, info_c, ctx)
    _assert_same_view(py, cpp, ctx)

    done = False
    while not done:
        legal = py.legal_actions()
        pawn = [a for a in legal if a < NUM_PAWN_ACTIONS]
        if pawn and rng.random() < pawn_bias:
            action = rng.choice(pawn)
        else:
            action = rng.choice(legal)
        _, term, trunc, info = _step_both(
            py, cpp, action, f'{ctx} ply={py.move_count}')
        done = term or trunc
    return info['winner']


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def _test_constructor_invariants():
    for n, walls in ((3, 2), (5, 3), (7, 5), (9, 10)):
        py, cpp = Quoridor(n, walls), QuoridorCpp(n, walls)
        assert cpp.n == py.n == n
        assert cpp.max_walls == py.max_walls == walls
        assert cpp.max_moves == py.max_moves == 12 * n * n // 5
        assert cpp.num_actions == py.num_actions
        assert cpp.obs_dim == py.obs_dim
        _assert_same_view(py, cpp, f'initial n={n}')
    assert QuoridorCpp(5, 3, max_moves=17).max_moves == 17

    for bad_ctor in (lambda: Quoridor(2, 3), lambda: QuoridorCpp(2, 3)):
        try:
            bad_ctor()
        except ValueError:
            pass
        else:
            raise AssertionError('board_size=2 did not raise ValueError')


def _test_jump_positions():
    # The jump scenarios from the Python self-tests, replayed in lockstep
    # (the Python side already asserts the rules; here we assert agreement).
    for kwargs, actions in (
        (dict(p1=(3, 2), p2=(2, 2)), [4]),                         # straight
        (dict(p1=(3, 2), p2=(2, 2), hw=[(2, 2)]), [2]),            # blocked
        (dict(p1=(3, 2), p2=(2, 2), hw=[(1, 2)]), [8]),            # diag NE
        (dict(p1=(3, 2), p2=(2, 2), hw=[(1, 2)], vw=[(1, 2)]), [9]),
        (dict(p1=(1, 2), p2=(0, 2)), [9]),                         # edge diag
        (dict(p1=(2, 2), p2=(3, 2), to_play=-1), [4]),             # O jumps
    ):
        py, cpp = _pair(**kwargs)
        for action in actions:
            _step_both(py, cpp, action, f'jump {kwargs}')


def _test_wall_rule_positions():
    for kwargs in (
        dict(p1=(4, 2), p2=(0, 2), hw=[(2, 2)]),      # overlap / crossing
        dict(p1=(4, 2), p2=(0, 2), vw=[(2, 2)]),
        dict(p1=(4, 0), p2=(0, 2), vw=[(3, 0)]),      # seal-path pocket
        dict(p1=(4, 2), p2=(0, 2), walls=(0, 3)),     # walls exhausted
        dict(p1=(4, 2), p2=(0, 2), walls=(0, 3), to_play=-1),
        dict(p1=(3, 1), p2=(1, 3), to_play=-1,        # mirrored mixed state
             hw=[(3, 0)], vw=[(1, 3)], walls=(1, 2)),
    ):
        _pair(**kwargs)  # _pair itself asserts identical legal walls

    # Playing the path-sealing wall anyway is an illegal-move loss in both.
    py, cpp = _pair(p1=(4, 0), p2=(0, 2), vw=[(3, 0)])
    blocking = wall_action('H', 2, 0, 5)
    assert blocking not in py.legal_actions()
    r, term, trunc, info = _step_both(py, cpp, blocking, 'seal path')
    assert r == -1.0 and term and info['winner'] == -1


def _test_wins_both_players():
    py, cpp = _pair(p1=(1, 2), p2=(3, 2))
    r, term, _, info = _step_both(py, cpp, 0, 'X wins')
    assert r == 1.0 and term and info['winner'] == 1

    py, cpp = _pair(p1=(1, 2), p2=(3, 2), to_play=-1)
    r, term, _, info = _step_both(py, cpp, 0, 'O wins')
    assert r == 1.0 and term and info['winner'] == -1

    # A wall from next to the goal row does not win in either backend.
    py, cpp = _pair(p1=(1, 2), p2=(3, 2))
    r, term, trunc, _ = _step_both(py, cpp, wall_action('H', 0, 0, 5), 'wall')
    assert r == 0.0 and not term and not trunc


def _test_illegal_and_exception_parity():
    # Illegal action: immediate loss with identical observables.
    py, cpp = Quoridor(5, 3), QuoridorCpp(5, 3)
    py.reset(), cpp.reset()
    r, term, _, info = _step_both(py, cpp, 1, 'illegal S')  # off the board
    assert r == -1.0 and term and info['winner'] == -1

    # step() on a finished game raises RuntimeError in both.
    for env in (py, cpp):
        try:
            env.step(0)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f'{type(env).__name__}: no RuntimeError')

    # Out-of-range action raises ValueError in both.
    for env in (Quoridor(5, 3), QuoridorCpp(5, 3)):
        try:
            env.step(44)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{type(env).__name__}: no ValueError')

    # set_state validates identically in both.
    for env in (Quoridor(5, 3), QuoridorCpp(5, 3)):
        for bad in (
            dict(p1=(2, 2), p2=(2, 2)),                # pawns collide
            dict(p1=(5, 0), p2=(0, 0)),                # off the board
            dict(p1=(4, 0), p2=(0, 0), to_play=2),     # bad player
            dict(p1=(4, 0), p2=(0, 0), h_walls=[(4, 0)]),  # bad wall slot
        ):
            kwargs = dict(p1=(4, 2), p2=(0, 2), to_play=1, h_walls=(),
                          v_walls=(), walls_p1=3, walls_p2=3)
            kwargs.update(bad)
            try:
                env.set_state(**kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f'{type(env).__name__}: set_state({bad}) did not raise')


def _test_truncation_parity():
    py = Quoridor(5, 3, max_moves=4)
    cpp = QuoridorCpp(5, 3, max_moves=4)
    py.reset(), cpp.reset()
    for action in (2, 2, 3, 3):
        r, term, trunc, info = _step_both(py, cpp, action, 'truncation')
    assert trunc and not term and info['winner'] == 0
    assert py.legal_actions() == cpp.legal_actions() == []


def _test_clone_independence():
    from copy import deepcopy
    base = QuoridorCpp(5, 3)
    base.reset()
    base.step(0)
    for copy_fn in (QuoridorCpp.clone, deepcopy):
        c = copy_fn(base)
        c.step(wall_action('H', 1, 0, 5))
        assert base.h_walls == set() and len(c.h_walls) == 1
        assert base.walls_left[-1] == 3 and c.walls_left[-1] == 2
        assert base.to_play == -1 and c.to_play == 1
        c.step(0)
        assert base.pos[1] == (3, 2) and c.pos[1] == (2, 2)


def _test_random_lockstep_5x5():
    rng = random.Random(0)
    winners = [_lockstep_game(rng, 5, 3, ctx=f'5x5 game={i}')
               for i in range(200)]
    assert any(w != 0 for w in winners)


def _test_random_lockstep_jump_heavy():
    # Pawn-biased policies keep the pawns adjacent, hammering the jump
    # rules (straight, diagonal, edge) far more often than uniform play.
    rng = random.Random(1)
    for i in range(60):
        _lockstep_game(rng, 5, 3, pawn_bias=0.85, ctx=f'jumpy game={i}')
    for i in range(30):
        _lockstep_game(rng, 3, 2, pawn_bias=0.6, ctx=f'3x3 game={i}')


def _test_random_lockstep_9x9():
    rng = random.Random(2)
    for i in range(8):
        _lockstep_game(rng, 9, 10, pawn_bias=0.5, ctx=f'9x9 game={i}')


# ---------------------------------------------------------------------------
# benchmark: python -m alphazero.test_cpp_backend --bench
# ---------------------------------------------------------------------------

def _bench():
    import time

    def run(env_cls, n, walls, games, seed):
        rng = random.Random(seed)
        plies = 0
        start = time.perf_counter()
        for _ in range(games):
            env = env_cls(n, walls)
            env.reset()
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(rng.choice(env.legal_actions()))
                plies += 1
                done = term or trunc
        return time.perf_counter() - start, plies

    for n, walls, games in ((5, 3, 100), (9, 10, 20)):
        t_py, plies = run(Quoridor, n, walls, games, seed=42)
        t_cpp, _ = run(QuoridorCpp, n, walls, games, seed=42)
        print(f'{n}x{n}: {games} random games, {plies} plies | '
              f'python {t_py:.2f}s, cpp {t_cpp:.2f}s '
              f'({t_py / t_cpp:.1f}x faster, '
              f'{plies / t_cpp / 1000:.0f}k plies/s)')


if __name__ == '__main__':
    import sys

    tests = [
        _test_constructor_invariants,
        _test_jump_positions,
        _test_wall_rule_positions,
        _test_wins_both_players,
        _test_illegal_and_exception_parity,
        _test_truncation_parity,
        _test_clone_independence,
        _test_random_lockstep_5x5,
        _test_random_lockstep_jump_heavy,
        _test_random_lockstep_9x9,
    ]
    for test in tests:
        test()
        print(f'{test.__name__}: ok')
    print('all tests passed')
    if '--bench' in sys.argv:
        _bench()
