from math import sqrt
from copy import deepcopy
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from game_config import GAME_ACTIONS
from nn import PolicyV, PolicyP

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

# tunable constants
c_puct = 1.0            # exploration weight in the PUCT formula
MCTS_POLICY_EXPLORE = 100  # number of simulations per move


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
        self.game = game # environment
        self.observation = observation # canonical observation (player to move)
        self.done = done # whether the game has concluded
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
        Create one child node per LEGAL action by applying that action to a
        copy of this node's environment.  Terminal children capture the real
        game outcome so the search backs up true results, not estimates.
        """
        if self.done:
            return

        child = {}
        for action in self.game.legal_actions():
            new_game = deepcopy(self.game)
            observation, reward, terminated, truncated, info = new_game.step(action)
            done = terminated or truncated
            # reward is from the mover's perspective; the child's perspective
            # is the opponent's, hence the sign flip.
            terminal_value = -reward if done else 0.0
            child[action] = Node(new_game, done, self, observation, action,
                                 terminal_value)

        self.child = child

    def rollout(self):
        """
        Evaluate this node.

        * Terminal node: return the TRUE game outcome (from the perspective of
          the player to move here) — no network estimate can beat the ground
          truth, and repeatedly visiting a terminal node keeps reinforcing it.
        * Otherwise: value from the value network, and policy priors from the
          policy network masked to legal actions and renormalised.
        """
        if self.done:
            return self.terminal_value, None

        obs = np.array(self.observation, dtype=np.float64).reshape(1, -1)

        v = policy_v(obs).numpy().flatten()[0]
        p = policy_p(obs).numpy().flatten()

        mask = np.zeros(GAME_ACTIONS, dtype=np.float64)
        mask[self.game.legal_actions()] = 1.0
        p = p * mask
        total = p.sum()
        if total > 0:
            p = p / total
        else:
            # network put all its mass on illegal moves: fall back to uniform
            p = mask / mask.sum()

        return v, p

    def explore(self):
        """
        Run one MCTS simulation:
            - from the current node, recursively pick the children which maximize PUCT
            - when a leaf is reached:
                - if it has never been evaluated, evaluate it with the network
                - otherwise expand it (create children) and evaluate one child
            - backpropagate the leaf value up to the root, flipping its sign at
              every level (negamax) so each node accumulates values from its
              own player's perspective
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

        # --- expansion / evaluation ---
        if current.N < 1:
            current.nn_v, current.nn_p = current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(list(current.child.values()))
            current.nn_v, current.nn_p = current.rollout()

        # --- backpropagation (negamax) ---
        value = current.nn_v  # perspective of the player to move at `current`
        node = current
        node.T = node.T + value
        node.N += 1
        while node.parent:
            node = node.parent
            value = -value
            node.T = node.T + value
            node.N += 1

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

        return next_child, action, next_child.observation, probs, self.observation

    def detach_parent(self):
        del self.parent
        self.parent = None


def Policy_Player_MCTS(mytree, greedy=False):
    """
    Core of AlphaZero move selection:
    * run MCTS_POLICY_EXPLORE simulations from the current node to gather statistics
    * pick the next action (sampled by visit count, or greedily if requested)

    Returns the sub-tree rooted at the chosen action (detached from its parent
    so that the statistics gathered so far are reused on the next move).
    Because node values are negamax (relative to the player to move at each
    node), the same tree is valid for both players in self-play.
    """
    for _ in range(MCTS_POLICY_EXPLORE):
        mytree.explore()

    next_tree, next_action, obs, p, p_obs = mytree.next(greedy=greedy)

    next_tree.detach_parent()

    return next_tree, next_action, obs, p, p_obs
