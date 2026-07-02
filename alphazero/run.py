"""
AlphaZero training driver (two-player self-play).

Each episode plays one full game of self-play in which BOTH sides are moved
by the same MCTS + networks.  Every visited position is stored with:
  * the canonical observation (perspective of the player to move),
  * the MCTS visit-count policy as the policy target,
  * the final game outcome z in {+1, 0, -1} FROM THAT PLAYER'S PERSPECTIVE
    as the value target.

Self-play exploration (AlphaZero-style):
  * Dirichlet noise is mixed into the root priors of every search;
  * moves are sampled by visit count (temperature 1) for the first
    TEMP_MOVES plies, then played greedily, so long games don't degenerate
    into random walks while the opening still gets explored.

Run:  python -m alphazero.run
Outputs: training plots (*.png) and network weights under checkpoints/.
"""

from copy import deepcopy
import os
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alphazero.game_config import make_game, GAME_NAME
from alphazero.replay_buffer import ReplayBuffer
# Import the SAME network instances the MCTS uses, so training the networks
# actually improves self-play.
from alphazero.mcts import Node, Policy_Player_MCTS, policy_v, policy_p


BUFFER_SIZE = 3000
BATCH_SIZE = 128
TRAIN_BATCHES_PER_EPISODE = 4  # Quoridor games yield ~10-60 positions each

EPISODES = 300
TEMP_MOVES = 8       # plies sampled by visit count before turning greedy
EVAL_EVERY = 50      # evaluate against a random player every N episodes
EVAL_GAMES = 20

METRIC_WINDOW = 50   # trailing window for the progress metrics


def self_play_episode(replay_buffer):
    """
    Play one self-play game and fill the replay buffer with
    (obs, outcome-for-that-player, mcts-policy) triples.
    Returns (winner, number of moves).
    """
    game = make_game()
    observation, info = game.reset()

    mytree = Node(deepcopy(game), False, None, observation, None)

    observations = []   # canonical obs of each position where MCTS ran
    policies = []       # MCTS visit-count policy at that position
    players = []        # player to move at that position (+1 / -1)

    done = False
    while not done:
        players.append(game.to_play)

        # p    = MCTS visit-count policy at the current position
        # p_ob = canonical observation of the current position
        greedy = len(players) > TEMP_MOVES
        mytree, action, _, p, p_ob = Policy_Player_MCTS(
            mytree, greedy=greedy, root_noise=True)

        observations.append(p_ob)
        policies.append(p)

        _, _, terminated, truncated, info = game.step(action)
        done = terminated or truncated

    winner = info['winner']  # +1 / -1 / 0

    for ob, p, player in zip(observations, policies, players):
        # Outcome from the perspective of the player to move at this
        # position: +1 if they went on to win, -1 to lose, 0 draw.
        replay_buffer.add(obs=ob, v=float(winner * player), p=p)

    return winner, len(players)


def train_networks(replay_buffer):
    """One gradient step per network on a sampled batch. Returns the losses."""
    experiences = replay_buffer.sample()

    inputs = np.array([exp.obs for exp in experiences], dtype=np.float64)

    v_targets = np.array([[exp.v] for exp in experiences], dtype=np.float64)
    loss_v = policy_v.train_on_batch(inputs, v_targets)

    p_targets = np.array([exp.p for exp in experiences], dtype=np.float64)
    loss_p = policy_p.train_on_batch(inputs, p_targets)

    # train_on_batch returns [loss, metric] when metrics are compiled
    return float(np.ravel(loss_v)[0]), float(np.ravel(loss_p)[0])


def evaluate_vs_random(games=EVAL_GAMES):
    """
    Play greedy (most-visited-move) MCTS against a uniform-random player,
    alternating who goes first.  Returns counts from the bot's perspective.
    """
    results = {'win': 0, 'draw': 0, 'loss': 0}

    for g in range(games):
        bot = 1 if g % 2 == 0 else -1
        game = make_game()
        observation, info = game.reset()

        done = False
        while not done:
            if game.to_play == bot:
                tree = Node(deepcopy(game), False, None, observation, None)
                _, action, _, _, _ = Policy_Player_MCTS(tree, greedy=True)
            else:
                action = random.choice(game.legal_actions())

            observation, _, terminated, truncated, info = game.step(action)
            done = terminated or truncated

        z = info['winner'] * bot
        results['win' if z > 0 else 'loss' if z < 0 else 'draw'] += 1

    return results


def train():
    replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    outcomes = []        # winner of each self-play game (+1 / -1 / 0)
    game_lengths = []    # plies per self-play game
    p1_win_rates = []    # trailing share of games won by the first player
    draw_rates = []      # trailing share of drawn (truncated) games
    v_losses = []
    p_losses = []
    eval_history = []    # (episode, results dict)

    for e in range(EPISODES):
        winner, moves = self_play_episode(replay_buffer)
        outcomes.append(winner)
        game_lengths.append(moves)

        window = outcomes[-METRIC_WINDOW:]
        p1_win_rates.append(window.count(1) / len(window))
        draw_rates.append(window.count(0) / len(window))

        print(f'episode {e + 1}: winner {winner:+d} in {moves} moves, '
              f'last {len(window)}: '
              f'P1 {p1_win_rates[-1]:.2f} / draw {draw_rates[-1]:.2f}, '
              f'avg length {np.mean(game_lengths[-METRIC_WINDOW:]):.1f}',
              flush=True)

        if len(replay_buffer) > BATCH_SIZE:
            for _ in range(TRAIN_BATCHES_PER_EPISODE):
                loss_v, loss_p = train_networks(replay_buffer)
            v_losses.append(loss_v)
            p_losses.append(loss_p)

        if (e + 1) % EVAL_EVERY == 0:
            results = evaluate_vs_random()
            eval_history.append((e + 1, results))
            print(f'--- eval vs random after episode {e + 1}: {results}',
                  flush=True)

    # persist the trained networks
    os.makedirs('checkpoints', exist_ok=True)
    policy_v.save_weights('checkpoints/policy_v.weights.h5')
    policy_p.save_weights('checkpoints/policy_p.weights.h5')

    # plots (rendered once at the end so the training loop never blocks)
    plt.figure()
    plt.plot(game_lengths, alpha=0.3, label='per game')
    if len(game_lengths) >= METRIC_WINDOW:
        smooth = np.convolve(game_lengths,
                             np.ones(METRIC_WINDOW) / METRIC_WINDOW,
                             mode='valid')
        plt.plot(range(METRIC_WINDOW - 1, len(game_lengths)), smooth,
                 label=f'trailing {METRIC_WINDOW}')
    plt.legend()
    plt.title(f'{GAME_NAME} self-play game length')
    plt.savefig('game_length.png')

    plt.figure()
    plt.plot(p1_win_rates, label='P1 win rate')
    plt.plot(draw_rates, label='draw rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f'self-play outcomes (trailing {METRIC_WINDOW} episodes)')
    plt.savefig('win_rate.png')

    plt.figure()
    plt.plot(v_losses)
    plt.title('value loss')
    plt.savefig('value_loss.png')

    plt.figure()
    plt.plot(p_losses)
    plt.title('policy loss')
    plt.savefig('policy_loss.png')

    if eval_history:
        print('\neval-vs-random history:')
        for episode, results in eval_history:
            print(f'  episode {episode}: {results}')


if __name__ == '__main__':
    train()
