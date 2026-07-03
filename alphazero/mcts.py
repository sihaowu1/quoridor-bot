from math import sqrt
from copy import deepcopy
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from alphazero.game_config import GAME_ACTIONS, GAME_OBS
from alphazero.nn import PolicyV, PolicyP

# ---------------------------------------------------------------------------
# Shared networks.
#
# These are the ONE pair of networks used both by the MCTS rollouts (below) and
# by the training driver in run.py (which imports these same objects).  Keeping
# a single instance is what makes self-play actually improve: gradient updates
# land on the exact networks the search uses to evaluate leaves.
#
# Because observations are canonical (always from the perspective of the
# player to move), a single value network serves both players: it always
# answers "how good is this position for whoever moves next?".
# ---------------------------------------------------------------------------
policy_v = PolicyV()
policy_v.compile(optimizer=keras.optimizers.Adam(),
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanSquaredError()])

policy_p = PolicyP()
policy_p.compile(optimizer=keras.optimizers.Adam(),
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=[tf.keras.metrics.CategoricalCrossentropy()])

# Single compiled graph for leaf evaluation.  Calling the Keras models
# directly from Python (`policy_v(obs)`) rebuilds the eager call machinery on
# every invocation; wrapping both forward passes in one tf.function with a
# fixed input signature makes each call a single graph execution, and the
# `None` batch dimension lets the same graph serve any batch size (1 during
# evaluation, PARALLEL_GAMES during batched self-play) without retracing.
predict_fn = tf.function(
    lambda x: (policy_v(x), policy_p(x)),
    input_signature=[tf.TensorSpec(shape=(None, GAME_OBS), dtype=tf.float32)],
)

# tunable constants
c_puct = 3.0          # exploration weight in the PUCT formula
MCTS_POLICY_EXPLORE = 800  # number of simulations per move
DIRICHLET_FRACTION = 0.25  # weight of the root exploration noise
DIRICHLET_ALPHA_SCALE = 10.0  # alpha = scale / num_legal_actions
# Self-play games run in lock-step so their leaf evaluations share one
# batched network call (32-128 is a reasonable range; override with the
# AZ_PARALLEL_GAMES environment variable).
PARALLEL_GAMES = int(os.environ.get('AZ_PARALLEL_GAMES', '64'))


class Node:
    """
    Node of the MCTS tree.

    Sign convention (negamax): every value stored on a node — ``T``, ``nn_v``,
    ``terminal_value`` — is from the perspective of the player TO MOVE at that
    node.  A parent therefore reads its child's mean value as ``-T/N``: what is
    good for the side to move at the child is exactly that bad for the side
    who chose the move at the parent.

    For a terminal node "the player to move" is the player who did NOT make
    the final move, so ``terminal_value`` is the negated reward of the mover:
    -1 below a winning move, 0 below a drawing one.
    """

    def __init__(self, game, done, parent, observation, action_index,
                 terminal_value=0.0):
        self.child = None # dict {action: Node}, only legal actions
        self.T = 0 # total value backed up through this node
        self.N = 0 # visit count
        self.game = game # environment (None on an unvisited lazy stub)
        self.observation = observation # canonical observation (player to move)
        self.done = done # whether the game has concluded (stubs: unknown
                         # until materialize(); False placeholder)
        self.parent = parent # link to parent node for backpropagation
        self.action_index = action_index # action that leads to this node
        self.terminal_value = terminal_value # true game outcome if done

        self.nn_v = 0 # value of the node according to the value network
        self.nn_p = None # move priors according to the policy network

    def get_UCB_score(self):
        """
        AlphaZero PUCT score used by the parent to pick children:

            U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))

        Q(s,a) is the child's mean value NEGATED, because the child stores its
        statistics from its own player's perspective (see class docstring).

        The prior P(s,a) (from the policy network) guides exploration of
        *unvisited* children too, so we deliberately do not short-circuit
        N == 0 with +inf the way a vanilla UCT implementation would.
        """
        q_score = -(self.T / self.N) if self.N > 0 else 0.0

        prior = self.parent.nn_p[self.action_index]
        u_score = c_puct * prior * sqrt(self.parent.N) / (1 + self.N)

        return q_score + u_score

    def create_child(self):
        """
        Create one STUB child node per LEGAL action.

        Lazy expansion: a stub carries only the action and the parent link —
        no environment copy, no observation.  The PUCT selection over fresh
        children needs neither (priors live in the parent's ``nn_p``), and
        most siblings are never visited at all, so cloning the environment
        for every legal action here would waste almost all of the copies.
        The clone + step happen in ``materialize()``, once, when a child is
        first stepped into.
        """
        if self.done:
            return

        self.child = {action: Node(None, False, self, None, action)
                      for action in self.game.legal_actions()}

    def materialize(self):
        """
        Fill in a stub child on first visit: clone the parent's environment
        once and apply this node's action, setting ``game``, ``observation``,
        ``done`` and ``terminal_value``.  Terminal nodes capture the real
        game outcome so the search backs up true results, not estimates.
        No-op on nodes that already carry an environment (e.g. roots).
        """
        if self.game is not None:
            return

        game = deepcopy(self.parent.game)
        observation, reward, terminated, truncated, info = \
            game.step(self.action_index)
        self.game = game
        self.observation = observation
        self.done = terminated or truncated
        # reward is from the mover's perspective; this node's perspective
        # is the opponent's, hence the sign flip.
        self.terminal_value = -reward if self.done else 0.0

    def select_leaf(self):
        """
        Selection + expansion half of one MCTS simulation: descend from this
        node picking the PUCT-maximising child at every level; when a leaf is
        reached that has already been evaluated (N >= 1), expand it and step
        into a random fresh child.  Returns the node awaiting evaluation.
        """
        current = self

        # --- selection ---
        while current.child:
            child = current.child
            scores = {a: node.get_UCB_score() for a, node in child.items()}
            max_U = max(scores.values())
            actions = [a for a, score in scores.items() if score == max_U]

            action = random.choice(actions)
            current = child[action]

        # --- expansion ---
        # The leaf may be a lazy stub (created by create_child, never
        # visited): give it its environment now — its own expansion and
        # evaluation both need it.
        current.materialize()

        if current.N >= 1 and not current.done:
            current.create_child()
            if current.child:
                current = random.choice(list(current.child.values()))
                current.materialize()

        return current

    def backpropagate(self):
        """
        Backpropagation half of one MCTS simulation: push this node's
        evaluated value (``nn_v``) up to the root, flipping its sign at every
        level (negamax) so each node accumulates values from its own player's
        perspective.
        """
        value = self.nn_v  # perspective of the player to move at `self`
        node = self
        node.T = node.T + value
        node.N += 1
        while node.parent:
            node = node.parent
            value = -value
            node.T = node.T + value
            node.N += 1

    def explore(self):
        """Run one full MCTS simulation on this tree alone (batch of 1)."""
        explore_batch([self])

    def next(self, greedy=False):
        """
        Pick the next action after the tree search.

        * greedy=False: sample proportional to the visit counts of the root's
          children (temperature = 1) — used during self-play for exploration.
        * greedy=True: play the most-visited move — used for evaluation.

        The returned policy target ``probs`` spans the FULL action space
        (length GAME_ACTIONS) with zeros on illegal actions, so it can be fed
        straight to the policy network as a training target.
        """
        if self.done:
            raise ValueError('Game has ended')

        if not self.child:
            raise ValueError('No children found')

        counts = np.zeros(GAME_ACTIONS, dtype=np.float64)
        for action, node in self.child.items():
            counts[action] = node.N

        total = counts.sum()
        if total == 0:
            legal = list(self.child.keys())
            probs = counts
            probs[legal] = 1.0 / len(legal)
        else:
            probs = counts / total

        if greedy:
            best = counts.max()
            action = random.choice([a for a in self.child if counts[a] == best])
        else:
            actions = list(self.child.keys())
            action = random.choices(actions,
                                    weights=[probs[a] for a in actions])[0]

        next_child = self.child[action]
        # A sampled child has virtually always been visited (hence
        # materialized), but a zero-visit stub can slip through the
        # total == 0 fallback above; it needs its environment before the
        # caller detaches it from this node.
        next_child.materialize()

        return next_child, action, next_child.observation, probs, self.observation

    def detach_parent(self):
        del self.parent
        self.parent = None


def _masked_priors(p, legal):
    """Mask raw policy output to the legal actions and renormalise."""
    mask = np.zeros(GAME_ACTIONS, dtype=np.float64)
    mask[legal] = 1.0
    p = p.astype(np.float64) * mask
    total = p.sum()
    if total > 0:
        return p / total
    # network put all its mass on illegal moves: fall back to uniform
    return mask / mask.sum()


def explore_batch(roots):
    """
    Run one MCTS simulation on each tree in ``roots``, sharing a single
    batched network call for all leaf evaluations.

    Each tree independently walks selection + expansion (pure Python, cheap);
    the resulting non-terminal leaves — one per tree, all distinct because
    the trees are disjoint — are evaluated together in one ``predict_fn``
    call, then every leaf value is backpropagated as usual.

    Terminal leaves keep their TRUE game outcome (from the perspective of
    the player to move there) — no network estimate can beat the ground
    truth, and repeatedly visiting a terminal node keeps reinforcing it.
    """
    leaves = [root.select_leaf() for root in roots]

    pending = [leaf for leaf in leaves if not leaf.done]
    if pending:
        obs = np.array([leaf.observation for leaf in pending],
                       dtype=np.float32)
        v_batch, p_batch = predict_fn(tf.constant(obs))
        v_batch = v_batch.numpy().reshape(-1)
        p_batch = p_batch.numpy()
        for leaf, v, p in zip(pending, v_batch, p_batch):
            leaf.nn_v = float(v)
            leaf.nn_p = _masked_priors(p, leaf.game.legal_actions())

    for leaf in leaves:
        if leaf.done:
            leaf.nn_v = leaf.terminal_value
        leaf.backpropagate()


def add_root_noise(node):
    """
    Mix Dirichlet noise into the root priors (self-play exploration, as in
    the AlphaZero paper): p = (1 - eps) * p + eps * Dir(alpha), applied only
    on the legal actions so the noised priors still sum to 1.
    """
    legal = node.game.legal_actions()
    if len(legal) < 2:
        return
    alpha = DIRICHLET_ALPHA_SCALE / len(legal)
    noise = np.random.dirichlet([alpha] * len(legal))
    p = node.nn_p.copy()
    p[legal] = (1 - DIRICHLET_FRACTION) * p[legal] + DIRICHLET_FRACTION * noise
    node.nn_p = p


def Policy_Player_MCTS_batch(trees, greedy=False, root_noise=False):
    """
    Core of AlphaZero move selection, over many independent games at once:
    * run MCTS_POLICY_EXPLORE simulations from each tree, in lock-step, so
      every simulation round shares one batched network call (explore_batch)
    * pick each game's next action (sampled by visit count, or greedily)

    ``greedy`` is either one bool for all trees or a sequence with one flag
    per tree (self-play games at different move counts switch to greedy play
    at different times).

    With root_noise=True (self-play only) Dirichlet noise is mixed into the
    root priors before the simulations, so the search keeps exploring moves
    the raw policy would dismiss.

    Returns one (next_tree, action, obs, p, p_obs) tuple per input tree.
    Each returned sub-tree is rooted at the chosen action and detached from
    its parent, so the statistics gathered so far are reused on the next
    move.  Because node values are negamax (relative to the player to move
    at each node), the same tree is valid for both players in self-play.
    """
    if isinstance(greedy, bool):
        greedy = [greedy] * len(trees)

    remaining = [MCTS_POLICY_EXPLORE] * len(trees)

    if root_noise:
        # A fresh root has no priors yet; its first simulation evaluates it
        # without descending, so noise applied after it still precedes every
        # child selection.  Roots carried over from the previous move keep
        # their priors and skip this step.
        fresh = [i for i, t in enumerate(trees)
                 if not t.done and t.nn_p is None]
        if fresh:
            explore_batch([trees[i] for i in fresh])
            for i in fresh:
                remaining[i] -= 1
        for t in trees:
            if not t.done:
                add_root_noise(t)

    for step in range(max(remaining)):
        explore_batch([t for t, r in zip(trees, remaining) if r > step])

    results = []
    for tree, g in zip(trees, greedy):
        next_tree, next_action, obs, p, p_obs = tree.next(greedy=g)
        next_tree.detach_parent()
        results.append((next_tree, next_action, obs, p, p_obs))

    return results


def Policy_Player_MCTS(mytree, greedy=False, root_noise=False):
    """Single-game convenience wrapper around Policy_Player_MCTS_batch
    (used for evaluation games and tests; batched network calls of size 1)."""
    return Policy_Player_MCTS_batch([mytree], greedy=greedy,
                                    root_noise=root_noise)[0]
