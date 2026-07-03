"""
Quoridor environment for the AlphaZero pipeline.

The API and two-player conventions mirror tic_tac_toe.py exactly, so this
game drops into the existing MCTS/training code unchanged:

* gymnasium-style ``reset() -> (obs, info)`` and
  ``step(action) -> (obs, reward, terminated, truncated, info)``;
* players are +1 (starts on the bottom row, aims for row 0) and -1
  (starts on the top row, aims for row N-1);
* the observation is always CANONICAL — from the perspective of the player
  about to move — so one network evaluates both players;
* ``step`` returns the reward from the perspective of the player who JUST
  moved: +1 for reaching the goal row, 0 otherwise, -1 for an illegal
  action (immediate loss).  Proper play masks with ``info['action_mask']``;
* ``info`` always carries ``to_play``, ``legal_actions``, ``action_mask``
  and, when the game is over, ``winner`` (+1 / -1 / 0 for a draw);
* games longer than ``max_moves`` plies are truncated as draws
  (``truncated=True``, ``winner=0``) so self-play cannot loop forever.

The board size and wall count are parameters; the classic game is
``Quoridor(9, 10)``, training defaults to a 5x5 board with 3 walls each
(see game_config.py).

CANONICAL FRAME — both observations AND actions are expressed in the frame
of the player to move, in which they always travel "north" toward row 0.
For player -1 the environment flips rows internally (pawn row r <-> N-1-r,
wall slot row r <-> N-2-r, N <-> S in pawn actions; columns are untouched
because Quoridor is left-right symmetric).  All translation happens inside
this class; the MCTS and networks never see absolute coordinates.

Action encoding (``num_actions(N) = 12 + 2*(N-1)^2``)::

    0-3   pawn step   N, S, E, W
    4-7   straight jump over the adjacent opponent  NN, SS, EE, WW
    8-11  diagonal jump (straight jump blocked)     NE, NW, SE, SW
    12 .. 12+(N-1)^2-1        horizontal wall at slot (r, c), row-major
    12+(N-1)^2 .. end         vertical   wall at slot (r, c), row-major

A wall slot (r, c) lies at the crossing between board rows r/r+1 and
columns c/c+1: a horizontal wall there blocks vertical movement between
rows r and r+1 for columns c and c+1; a vertical wall blocks horizontal
movement between columns c and c+1 for rows r and r+1.

Observation (``obs_size(N) = 2*N^2 + 2*(N-1)^2 + 2``): own-pawn one-hot,
opponent-pawn one-hot, horizontal-wall grid, vertical-wall grid, own and
opponent walls-left fractions.

Run ``python -m alphazero.quoridor`` to execute the self-tests.
"""

from collections import deque

import numpy as np

DEFAULT_BOARD_SIZE = 5
DEFAULT_WALLS = 3

NUM_PAWN_ACTIONS = 12

# Canonical deltas for actions 0-11 (see module docstring for the order).
PAWN_DELTAS = (
    (-1, 0), (1, 0), (0, 1), (0, -1),      # N, S, E, W
    (-2, 0), (2, 0), (0, 2), (0, -2),      # NN, SS, EE, WW
    (-1, 1), (-1, -1), (1, 1), (1, -1),    # NE, NW, SE, SW
)

_DIAG_ACTION = {(-1, 1): 8, (-1, -1): 9, (1, 1): 10, (1, -1): 11}


def num_actions(n):
    return NUM_PAWN_ACTIONS + 2 * (n - 1) ** 2


def obs_size(n):
    return 2 * n * n + 2 * (n - 1) ** 2 + 2


def wall_action(orientation, r, c, n):
    """Action index of the wall at slot (r, c) in the CANONICAL frame."""
    m = n - 1
    return NUM_PAWN_ACTIONS + (0 if orientation == 'H' else m * m) + r * m + c


def decode_action(action, n):
    """('move', delta) for pawn actions, (orientation, r, c) for walls."""
    if action < NUM_PAWN_ACTIONS:
        return ('move', PAWN_DELTAS[action])
    m = n - 1
    w = action - NUM_PAWN_ACTIONS
    if w < m * m:
        return ('H',) + divmod(w, m)
    return ('V',) + divmod(w - m * m, m)


def _step_blocked(r, c, dr, dc, hw, vw):
    """Wall between (r, c) and (r+dr, c+dc)? Single orthogonal steps only."""
    if dr == -1:
        return (r - 1, c) in hw or (r - 1, c - 1) in hw
    if dr == 1:
        return (r, c) in hw or (r, c - 1) in hw
    if dc == 1:
        return (r, c) in vw or (r - 1, c) in vw
    return (r, c - 1) in vw or (r - 1, c - 1) in vw


class Quoridor:

    def __init__(self, board_size=DEFAULT_BOARD_SIZE, walls=DEFAULT_WALLS,
                 max_moves=None):
        if board_size < 3:
            raise ValueError('board_size must be at least 3')
        self.n = board_size
        self.max_walls = walls
        # Long enough that real games never hit it; random early self-play
        # games that wander are cut off as draws.
        self.max_moves = max_moves or 800
        self.num_actions = num_actions(board_size)
        self.obs_dim = obs_size(board_size)
        self.reset()

    def reset(self, seed=None, options=None):
        n = self.n
        self.pos = {1: (n - 1, n // 2), -1: (0, n // 2)}
        self.walls_left = {1: self.max_walls, -1: self.max_walls}
        self.h_walls = set()   # absolute wall slots (r, c)
        self.v_walls = set()
        self.to_play = 1
        self.done = False
        self.winner = None     # +1 / -1 / 0 (draw), None while running
        self.move_count = 0
        self._legal_cache = None
        self._legal_set = None
        return self._obs(), self._info()

    # ------------------------------------------------------------------
    # public API used by the MCTS / training driver
    # ------------------------------------------------------------------

    def step(self, action):
        if self.done:
            raise RuntimeError('step() called on a finished game; call reset()')
        if not 0 <= action < self.num_actions:
            raise ValueError(
                f'action must be in [0, {self.num_actions}), got {action}')

        mover = self.to_play

        self.legal_actions()  # populate the cache / legal set
        if action not in self._legal_set:
            # Illegal move: immediate loss for the mover (same escape hatch
            # tic-tac-toe provides; masked play never triggers it).
            self.done = True
            self.winner = -mover
            return self._obs(), -1.0, True, False, self._info()

        won = False
        if action < NUM_PAWN_ACTIONS:
            dr, dc = PAWN_DELTAS[action]
            own = self._canonical_pos(mover)
            cr, cc = own[0] + dr, own[1] + dc     # canonical destination
            won = cr == 0                          # goal row in canonical frame
            self.pos[mover] = (cr, cc) if mover == 1 else (self.n - 1 - cr, cc)
        else:
            orientation, r, c = decode_action(action, self.n)
            ar = r if mover == 1 else self.n - 2 - r
            (self.h_walls if orientation == 'H' else self.v_walls).add((ar, c))
            self.walls_left[mover] -= 1

        self.move_count += 1
        self._legal_cache = None
        self._legal_set = None

        truncated = False
        reward = 0.0
        if won:
            self.done = True
            self.winner = mover
            reward = 1.0

        self.to_play = -mover

        if not self.done and self.move_count >= self.max_moves:
            self.done = True
            self.winner = 0
            truncated = True

        # Safety net: with jump rules a player can in rare contrived cases
        # have no move at all; treat that as a draw rather than crash.
        if not self.done and not self.legal_actions():
            self.done = True
            self.winner = 0
            truncated = True

        terminated = self.done and not truncated
        return self._obs(), reward, terminated, truncated, self._info()

    def legal_actions(self):
        """Legal actions for the player to move, in the CANONICAL frame."""
        if self.done:
            return []
        if self._legal_cache is None:
            own, opp, hw, vw = self._canonical_state(copy_walls=True)
            acts = self._pawn_actions(own, opp, hw, vw)
            if self.walls_left[self.to_play] > 0:
                acts += self._wall_actions(own, opp, hw, vw)
            acts.sort()
            self._legal_cache = acts
            self._legal_set = frozenset(acts)
        return self._legal_cache

    def set_state(self, p1, p2, to_play, h_walls, v_walls,
                  walls_p1, walls_p2):
        """Install an arbitrary position (testing / analysis).

        Same signature and validation as the C++ engine's set_state, so
        backend-agnostic code can position either implementation.
        """
        n, m = self.n, self.n - 1
        p1, p2 = tuple(p1), tuple(p2)
        for r, c in (p1, p2):
            if not (0 <= r < n and 0 <= c < n):
                raise ValueError('invalid pawn positions')
        if p1 == p2:
            raise ValueError('invalid pawn positions')
        if to_play not in (1, -1):
            raise ValueError('to_play must be +1 or -1')
        h_walls = {tuple(w) for w in h_walls}
        v_walls = {tuple(w) for w in v_walls}
        for r, c in h_walls | v_walls:
            if not (0 <= r < m and 0 <= c < m):
                raise ValueError('wall slot out of range')
        self.pos = {1: p1, -1: p2}
        self.to_play = to_play
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.walls_left = {1: walls_p1, -1: walls_p2}
        self.done = False
        self.winner = None
        self.move_count = 0
        self._legal_cache = None
        self._legal_set = None

    def clone(self):
        """Cheap copy for tree search (also backs __deepcopy__)."""
        c = Quoridor.__new__(Quoridor)
        c.n = self.n
        c.max_walls = self.max_walls
        c.max_moves = self.max_moves
        c.num_actions = self.num_actions
        c.obs_dim = self.obs_dim
        c.pos = dict(self.pos)
        c.walls_left = dict(self.walls_left)
        c.h_walls = set(self.h_walls)
        c.v_walls = set(self.v_walls)
        c.to_play = self.to_play
        c.done = self.done
        c.winner = self.winner
        c.move_count = self.move_count
        # The cached legal list/set are never mutated, so sharing is safe.
        c._legal_cache = self._legal_cache
        c._legal_set = self._legal_set
        return c

    def __deepcopy__(self, memo):
        return self.clone()

    def render(self):
        print(str(self))

    def __str__(self):
        n, hw, vw = self.n, self.h_walls, self.v_walls
        lines = []
        for r in range(n):
            row = []
            for c in range(n):
                ch = '.'
                if self.pos[1] == (r, c):
                    ch = 'X'
                if self.pos[-1] == (r, c):
                    ch = 'O'
                row.append(ch)
                if c < n - 1:
                    row.append('|' if (r, c) in vw or (r - 1, c) in vw else ' ')
            lines.append(''.join(row))
            if r < n - 1:
                sep = []
                for c in range(n):
                    sep.append('-' if (r, c) in hw or (r, c - 1) in hw else ' ')
                    if c < n - 1:
                        sep.append(' ')
                lines.append(''.join(sep))
        lines.append(f'walls left: X={self.walls_left[1]} '
                     f'O={self.walls_left[-1]}  '
                     f'to move: {"X" if self.to_play == 1 else "O"}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # canonical-frame helpers
    # ------------------------------------------------------------------

    def _canonical_pos(self, player):
        r, c = self.pos[player]
        return (r, c) if player == 1 else (self.n - 1 - r, c)

    def _canonical_state(self, copy_walls=False):
        """(own pos, opp pos, h_walls, v_walls) in the mover's frame."""
        n = self.n
        if self.to_play == 1:
            hw = set(self.h_walls) if copy_walls else self.h_walls
            vw = set(self.v_walls) if copy_walls else self.v_walls
            return self.pos[1], self.pos[-1], hw, vw
        own = (n - 1 - self.pos[-1][0], self.pos[-1][1])
        opp = (n - 1 - self.pos[1][0], self.pos[1][1])
        hw = {(n - 2 - r, c) for r, c in self.h_walls}
        vw = {(n - 2 - r, c) for r, c in self.v_walls}
        return own, opp, hw, vw

    # ------------------------------------------------------------------
    # move generation (everything below operates in the canonical frame)
    # ------------------------------------------------------------------

    def _pawn_actions(self, own, opp, hw, vw):
        n = self.n
        acts = []
        for a in range(4):
            dr, dc = PAWN_DELTAS[a]
            nr, nc = own[0] + dr, own[1] + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if _step_blocked(own[0], own[1], dr, dc, hw, vw):
                continue
            if (nr, nc) != opp:
                acts.append(a)
                continue
            # The opponent stands on the destination: jump rules.
            jr, jc = nr + dr, nc + dc
            if (0 <= jr < n and 0 <= jc < n
                    and not _step_blocked(nr, nc, dr, dc, hw, vw)):
                acts.append(4 + a)  # straight jump over the opponent
            else:
                # Straight jump blocked by a wall or the board edge: the two
                # squares diagonally beside the opponent become reachable.
                for pd in ((0, 1), (0, -1)) if dc == 0 else ((1, 0), (-1, 0)):
                    tr, tc = nr + pd[0], nc + pd[1]
                    if not (0 <= tr < n and 0 <= tc < n):
                        continue
                    if _step_blocked(nr, nc, pd[0], pd[1], hw, vw):
                        continue
                    acts.append(_DIAG_ACTION[(dr + pd[0], dc + pd[1])])
        return acts

    def _wall_actions(self, own, opp, hw, vw):
        m = self.n - 1
        # Lattice nodes covered by existing walls, used to skip the path
        # check for walls that cannot possibly complete a barrier.
        nodes = set()
        for r, c in hw:
            nodes.update(((r + 1, c), (r + 1, c + 1), (r + 1, c + 2)))
        for r, c in vw:
            nodes.update(((r, c + 1), (r + 1, c + 1), (r + 2, c + 1)))

        acts = []
        for r in range(m):
            for c in range(m):
                if self._wall_ok('H', r, c, hw, vw, nodes, own, opp):
                    acts.append(NUM_PAWN_ACTIONS + r * m + c)
                if self._wall_ok('V', r, c, hw, vw, nodes, own, opp):
                    acts.append(NUM_PAWN_ACTIONS + m * m + r * m + c)
        return acts

    def _wall_ok(self, orientation, r, c, hw, vw, nodes, own, opp):
        n = self.n
        if orientation == 'H':
            if (r, c) in hw or (r, c) in vw:          # occupied / crossing
                return False
            if (r, c - 1) in hw or (r, c + 1) in hw:  # collinear overlap
                return False
            e1, mid, e2 = (r + 1, c), (r + 1, c + 1), (r + 1, c + 2)
            e1_border, e2_border = c == 0, c + 2 == n
        else:
            if (r, c) in vw or (r, c) in hw:
                return False
            if (r - 1, c) in vw or (r + 1, c) in vw:
                return False
            e1, mid, e2 = (r, c + 1), (r + 1, c + 1), (r + 2, c + 1)
            e1_border, e2_border = r == 0, r + 2 == n

        # A wall can only cut off a path if it is part of a barrier that
        # connects two border points, which requires the new wall to hook
        # into the border/existing walls at TWO of its three lattice nodes.
        anchors = ((e1_border or e1 in nodes)
                   + (e2_border or e2 in nodes)
                   + (mid in nodes))
        if anchors < 2:
            return True

        target = hw if orientation == 'H' else vw
        target.add((r, c))
        ok = (self._path_exists(own, 0, hw, vw)
              and self._path_exists(opp, n - 1, hw, vw))
        target.discard((r, c))
        return ok

    def _path_exists(self, start, goal_row, hw, vw):
        n = self.n
        seen = bytearray(n * n)
        seen[start[0] * n + start[1]] = 1
        dq = deque([start])
        while dq:
            r, c = dq.popleft()
            if r == goal_row:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if seen[nr * n + nc]:
                    continue
                if _step_blocked(r, c, dr, dc, hw, vw):
                    continue
                seen[nr * n + nc] = 1
                dq.append((nr, nc))
        return False

    # ------------------------------------------------------------------
    # observation / info
    # ------------------------------------------------------------------

    def _obs(self):
        own, opp, hw, vw = self._canonical_state()
        n, m = self.n, self.n - 1
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        obs[own[0] * n + own[1]] = 1.0
        obs[n * n + opp[0] * n + opp[1]] = 1.0
        base = 2 * n * n
        for r, c in hw:
            obs[base + r * m + c] = 1.0
        base += m * m
        for r, c in vw:
            obs[base + r * m + c] = 1.0
        denom = max(self.max_walls, 1)
        obs[-2] = self.walls_left[self.to_play] / denom
        obs[-1] = self.walls_left[-self.to_play] / denom
        return obs

    def _info(self):
        legal = self.legal_actions()
        mask = np.zeros(self.num_actions, dtype=np.int8)
        mask[legal] = 1
        info = {
            'to_play': self.to_play,
            'legal_actions': list(legal),
            'action_mask': mask,
        }
        if self.done:
            info['winner'] = self.winner
        return info


# ---------------------------------------------------------------------------
# Self-tests: python -m alphazero.quoridor
# ---------------------------------------------------------------------------

def _make(p1, p2, to_play=1, hw=(), vw=(), walls=(3, 3), n=5, max_moves=None):
    """Build an arbitrary (assumed reachable) position for testing."""
    g = Quoridor(n, 3, max_moves)
    g.set_state(p1, p2, to_play, hw, vw, walls[0], walls[1])
    return g


def _walls_in(actions):
    return [a for a in actions if a >= NUM_PAWN_ACTIONS]


def _test_initial_position():
    g = Quoridor(5, 3)
    obs, info = g.reset()
    assert g.num_actions == 44 and g.obs_dim == 84
    assert obs.shape == (84,)
    # two pawn one-hots + two full wall fractions
    assert obs.sum() == 4.0 and obs[-2] == 1.0 and obs[-1] == 1.0
    assert info['to_play'] == 1
    # bottom row: N, E, W legal (S off the board), plus all 32 walls
    legal = info['legal_actions']
    assert [a for a in legal if a < NUM_PAWN_ACTIONS] == [0, 2, 3]
    assert len(_walls_in(legal)) == 32 and len(legal) == 35
    assert info['action_mask'].sum() == 35


def _test_pawn_movement_and_wall_blocking():
    g = Quoridor(5, 3)
    g.reset()
    obs, reward, terminated, truncated, info = g.step(0)  # X: (4,2) -> (3,2)
    assert g.pos[1] == (3, 2) and reward == 0.0 and not terminated
    assert info['to_play'] == -1

    # O places a wall (canonical H(1,1) for O = absolute H(2,1)) that blocks
    # X's step from (3,2) to (2,2).
    obs, _, _, _, info = g.step(wall_action('H', 1, 1, 5))
    assert (2, 1) in g.h_walls and g.walls_left[-1] == 2
    assert 0 not in info['legal_actions']  # X can no longer step north


def _test_straight_jump():
    g = _make(p1=(3, 2), p2=(2, 2))
    legal = g.legal_actions()
    assert 0 not in legal      # destination occupied by the opponent
    assert 4 in legal          # straight jump over it
    assert 8 not in legal and 9 not in legal  # no diagonals: jump available
    g.step(4)
    assert g.pos[1] == (1, 2)

    # A wall between the pawns blocks both the step and the jump.
    g = _make(p1=(3, 2), p2=(2, 2), hw=[(2, 2)])
    legal = g.legal_actions()
    assert 0 not in legal and 4 not in legal
    assert 8 not in legal and 9 not in legal


def _test_diagonal_jump_wall_behind():
    # Wall behind the opponent (absolute H(1,2) blocks (2,2)-(1,2)):
    # straight jump illegal, both diagonals open.
    g = _make(p1=(3, 2), p2=(2, 2), hw=[(1, 2)])
    legal = g.legal_actions()
    assert 4 not in legal
    assert 8 in legal and 9 in legal
    g.step(8)                      # NE
    assert g.pos[1] == (2, 3)

    # Side wall V(1,2) additionally blocks (2,2)-(2,3): only NW remains.
    g = _make(p1=(3, 2), p2=(2, 2), hw=[(1, 2)], vw=[(1, 2)])
    legal = g.legal_actions()
    assert 8 not in legal and 9 in legal


def _test_diagonal_jump_edge_behind():
    # Opponent on the top edge: straight jump would leave the board.
    g = _make(p1=(1, 2), p2=(0, 2))
    legal = g.legal_actions()
    assert 4 not in legal
    assert 8 in legal and 9 in legal
    obs, reward, terminated, _, info = g.step(9)  # NW to (0,1): goal row
    assert terminated and reward == 1.0 and info['winner'] == 1


def _test_wall_occupancy_rules():
    g = _make(p1=(4, 2), p2=(0, 2), hw=[(2, 2)])
    legal = set(g.legal_actions())
    assert wall_action('H', 2, 2, 5) not in legal   # same slot
    assert wall_action('V', 2, 2, 5) not in legal   # crossing
    assert wall_action('H', 2, 1, 5) not in legal   # collinear overlap left
    assert wall_action('H', 2, 3, 5) not in legal   # collinear overlap right
    assert wall_action('H', 1, 2, 5) in legal       # parallel above is fine
    assert wall_action('V', 1, 2, 5) in legal       # T-junction is fine

    g = _make(p1=(4, 2), p2=(0, 2), vw=[(2, 2)])
    legal = set(g.legal_actions())
    assert wall_action('V', 2, 2, 5) not in legal
    assert wall_action('H', 2, 2, 5) not in legal   # crossing
    assert wall_action('V', 1, 2, 5) not in legal   # collinear overlap above
    assert wall_action('V', 3, 2, 5) not in legal   # collinear overlap below
    assert wall_action('V', 2, 1, 5) in legal


def _test_wall_cannot_seal_path():
    # V(3,0) blocks (3,0)-(3,1) and (4,0)-(4,1); H(2,0) would seal X (at
    # (4,0)) into the two-cell pocket {(3,0), (4,0)} with no way to row 0.
    g = _make(p1=(4, 0), p2=(0, 2), vw=[(3, 0)])
    blocking = wall_action('H', 2, 0, 5)
    assert blocking not in g.legal_actions()
    # ... and playing it anyway is an illegal-move loss for the mover.
    obs, reward, terminated, _, info = g.step(blocking)
    assert terminated and reward == -1.0 and info['winner'] == -1

    # The same wall is fine when X is not in the pocket.
    g = _make(p1=(4, 2), p2=(0, 2), vw=[(3, 0)])
    assert blocking in g.legal_actions()


def _test_walls_exhausted():
    g = _make(p1=(4, 2), p2=(0, 2), walls=(0, 3))
    assert _walls_in(g.legal_actions()) == []
    g = _make(p1=(4, 2), p2=(0, 2), walls=(0, 3), to_play=-1)
    assert len(_walls_in(g.legal_actions())) == 32


def _test_win_both_players():
    g = _make(p1=(1, 2), p2=(3, 2))
    obs, reward, terminated, _, info = g.step(0)
    assert terminated and reward == 1.0 and info['winner'] == 1
    assert g.pos[1] == (0, 2)

    # Player -1 aims for row N-1; canonical N from (3, 2) is absolute S.
    g = _make(p1=(1, 2), p2=(3, 2), to_play=-1)
    obs, reward, terminated, _, info = g.step(0)
    assert terminated and reward == 1.0 and info['winner'] == -1
    assert g.pos[-1] == (4, 2)

    # Wall placements never win, even from the goal-row-adjacent square.
    g = _make(p1=(1, 2), p2=(3, 2))
    obs, reward, terminated, truncated, info = g.step(wall_action('H', 0, 0, 5))
    assert not terminated and not truncated and reward == 0.0


def _test_canonical_symmetry():
    # B is A mirrored (players swapped, rows flipped): the canonical
    # observation and legal actions must be identical, and stay identical
    # after both games play the same canonical action.
    a = _make(p1=(3, 1), p2=(1, 3), to_play=1,
              hw=[(0, 0)], vw=[(2, 3)], walls=(2, 1))
    b = _make(p1=(4 - 1, 3), p2=(4 - 3, 1), to_play=-1,
              hw=[(3 - 0, 0)], vw=[(3 - 2, 3)], walls=(1, 2))
    assert np.array_equal(a._obs(), b._obs())
    assert a.legal_actions() == b.legal_actions()

    for action in (0, wall_action('V', 1, 1, 5), 2):
        oa = a.step(action)
        ob = b.step(action)
        assert np.array_equal(oa[0], ob[0]), f'obs diverged after {action}'
        assert oa[1:4] == ob[1:4]
        assert a.legal_actions() == b.legal_actions()
    # the mirror relation between the absolute states is preserved
    assert a.pos[1] == (4 - b.pos[-1][0], b.pos[-1][1])
    assert a.pos[-1] == (4 - b.pos[1][0], b.pos[1][1])
    assert a.h_walls == {(3 - r, c) for r, c in b.h_walls}
    assert a.v_walls == {(3 - r, c) for r, c in b.v_walls}


def _test_clone_independence():
    from copy import deepcopy
    g = Quoridor(5, 3)
    g.reset()
    g.step(0)
    for copy_fn in (Quoridor.clone, deepcopy):
        c = copy_fn(g)
        c.step(wall_action('H', 1, 0, 5))  # absolute H(2,0): away from col 2
        assert g.h_walls == set() and len(c.h_walls) == 1
        assert g.walls_left[-1] == 3 and c.walls_left[-1] == 2
        assert g.to_play == -1 and c.to_play == 1
        c.step(0)
        assert g.pos[1] == (3, 2) and c.pos[1] == (2, 2)


def _test_truncation():
    g = Quoridor(5, 3, max_moves=4)
    g.reset()
    for i, action in enumerate([2, 2, 3, 3]):  # shuffle sideways
        obs, reward, terminated, truncated, info = g.step(action)
    assert truncated and not terminated and reward == 0.0
    assert info['winner'] == 0 and g.move_count == 4
    assert g.legal_actions() == []


def _test_illegal_and_finished():
    g = Quoridor(5, 3)
    g.reset()
    # stepping into the wall-free but occupied/absent square: action 1 (S)
    # is off the board from the start position -> illegal move loses.
    obs, reward, terminated, _, info = g.step(1)
    assert terminated and reward == -1.0 and info['winner'] == -1

    try:
        g.step(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError('step() on a finished game did not raise')

    try:
        Quoridor(5, 3).step(44)
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range action did not raise')


def _test_random_playthroughs():
    import random
    rng = random.Random(0)
    decisive = 0
    for _ in range(200):
        g = Quoridor(5, 3)
        obs, info = g.reset()
        done = False
        while not done:
            legal = g.legal_actions()
            assert legal, 'no legal actions in a running game'
            assert list(np.flatnonzero(info['action_mask'])) == legal
            obs, reward, terminated, truncated, info = g.step(rng.choice(legal))
            assert reward in (0.0, 1.0)
            done = terminated or truncated
        assert info['winner'] in (1, -1, 0)
        assert g.move_count <= g.max_moves
        if info['winner'] != 0:
            decisive += 1
            assert not truncated
    assert decisive > 0, 'random play never produced a decisive game'


if __name__ == '__main__':
    tests = [
        _test_initial_position,
        _test_pawn_movement_and_wall_blocking,
        _test_straight_jump,
        _test_diagonal_jump_wall_behind,
        _test_diagonal_jump_edge_behind,
        _test_wall_occupancy_rules,
        _test_wall_cannot_seal_path,
        _test_walls_exhausted,
        _test_win_both_players,
        _test_canonical_symmetry,
        _test_clone_independence,
        _test_truncation,
        _test_illegal_and_finished,
        _test_random_playthroughs,
    ]
    for test in tests:
        test()
        print(f'{test.__name__}: ok')
    print('all tests passed')
