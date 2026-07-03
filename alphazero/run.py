"""
AlphaZero training driver (two-player self-play), crash-safe.

Each episode plays one full game of self-play in which BOTH sides are moved
by the same MCTS + networks.  Episodes run PARALLEL_GAMES at a time in
lock-step so that the MCTS leaf evaluations of all in-flight games are
batched into single network calls (the GPU-efficiency win; see mcts.py).
Every visited position is stored with:
  * the canonical observation (perspective of the player to move),
  * the MCTS visit-count policy as the policy target,
  * the final game outcome z in {+1, 0, -1} FROM THAT PLAYER'S PERSPECTIVE
    as the value target.

Self-play exploration (AlphaZero-style):
  * Dirichlet noise is mixed into the root priors of every search;
  * moves are sampled by visit count (temperature 1) for the first
    TEMP_MOVES plies, then played greedily, so long games don't degenerate
    into random walks while the opening still gets explored.

Checkpointing / resume (built for Colab, where the runtime can die at any
moment): the complete training state — network weights, optimizer state,
replay buffer, metrics, RNG — is saved atomically every CHECKPOINT_EVERY
episodes, and ``train()`` automatically resumes from the latest good
checkpoint when one exists.  Point AZ_CHECKPOINT_DIR at persistent storage
(e.g. a mounted Google Drive folder) and simply re-run after a crash.

Environment variables:
  AZ_CHECKPOINT_DIR    where checkpoints, weights and plots go
                       (default: checkpoints/)
  AZ_CHECKPOINT_EVERY  save frequency in episodes (default: 1)
  AZ_EPISODES          total episodes for the run (default: 300)
  AZ_GAME              game selection, see game_config.py
  AZ_PARALLEL_GAMES    self-play games run in lock-step per batch so their
                       MCTS leaf evaluations share one batched network call
                       (default: 64, see mcts.PARALLEL_GAMES)

Run:  python -m alphazero.run
"""

from copy import deepcopy
import os
import random
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alphazero.game_config import make_game, GAME_NAME, GAME_ACTIONS, GAME_OBS
from alphazero.replay_buffer import ReplayBuffer
from alphazero.checkpoint import (save_checkpoint, load_checkpoint,
                                  restore_networks, restore_buffer,
                                  restore_rng)
# Import the SAME network instances the MCTS uses, so training the networks
# actually improves self-play.
from alphazero.mcts import (Node, Policy_Player_MCTS,
                            Policy_Player_MCTS_batch, PARALLEL_GAMES,
                            policy_v, policy_p)


BUFFER_SIZE = 50000
BATCH_SIZE = 128
TRAIN_BATCHES_PER_EPISODE = 2

EPISODES = int(os.environ.get('AZ_EPISODES', '300'))
TEMP_MOVES = 24      # plies sampled by visit count before turning greedy
EVAL_EVERY = 50      # evaluate against a random player every N episodes
EVAL_GAMES = 20

METRIC_WINDOW = 50   # trailing window for the progress metrics

# In-batch progress heartbeat: nothing else prints until every game in a
# parallel self-play batch finishes, which can take hours (longest game x
# 800 simulations per move), so log every N plies that the batch is alive.
LOG_EVERY_PLIES = 10

CHECKPOINT_DIR = os.environ.get('AZ_CHECKPOINT_DIR', 'checkpoints')
CHECKPOINT_EVERY = int(os.environ.get('AZ_CHECKPOINT_EVERY', '1'))
CHECKPOINT_FILE = f'{GAME_NAME}_train_state.pkl'

# Validated against a checkpoint before resuming, so state from one game /
# board size can never be silently loaded into another configuration.
CHECKPOINT_META = {
    'game': GAME_NAME,
    'actions': GAME_ACTIONS,
    'obs': GAME_OBS,
}


class _SelfPlayGame:
    """State of one in-flight self-play game (see self_play_episodes)."""

    def __init__(self):
        self.game = make_game()
        observation, info = self.game.reset()
        self.tree = Node(deepcopy(self.game), False, None, observation, None)

        self.observations = []  # canonical obs of each position where MCTS ran
        self.policies = []      # MCTS visit-count policy at that position
        self.players = []       # player to move at that position (+1 / -1)
        self.winner = None      # +1 / -1 / 0 once the game ends
        self.done = False


def self_play_episodes(replay_buffer, n_games=PARALLEL_GAMES):
    """
    Play ``n_games`` self-play games in parallel and fill the replay buffer
    with (obs, outcome-for-that-player, mcts-policy) triples.

    The games advance in lock-step: every simulation round batches the leaf
    evaluations of all still-running games into a single network call
    (Policy_Player_MCTS_batch), which is what keeps the GPU busy.  Finished
    games drop out of the batch; the last few moves therefore run at smaller
    batch sizes.

    Returns a list of (winner, number of moves), one entry per game.
    """
    games = [_SelfPlayGame() for _ in range(n_games)]

    start = time.time()
    ply = 0
    active = list(games)
    while active:
        ply += 1
        for g in active:
            g.players.append(g.game.to_play)

        # p    = MCTS visit-count policy at the current position
        # p_ob = canonical observation of the current position
        greedy = [len(g.players) > TEMP_MOVES for g in active]
        results = Policy_Player_MCTS_batch([g.tree for g in active],
                                           greedy=greedy, root_noise=True)

        for g, (tree, action, _, p, p_ob) in zip(active, results):
            g.tree = tree
            g.observations.append(p_ob)
            g.policies.append(p)

            _, _, terminated, truncated, info = g.game.step(action)
            if terminated or truncated:
                g.winner = info['winner']
                g.done = True

        active = [g for g in active if not g.done]

        if ply % LOG_EVERY_PLIES == 0 and active:
            elapsed = time.time() - start
            print(f'  self-play: ply {ply}, {len(active)}/{n_games} games '
                  f'running, {elapsed:.0f}s elapsed '
                  f'({elapsed / ply:.1f}s/ply)', flush=True)

    for g in games:
        for ob, p, player in zip(g.observations, g.policies, g.players):
            # Outcome from the perspective of the player to move at this
            # position: +1 if they went on to win, -1 to lose, 0 draw.
            replay_buffer.add(obs=ob, v=float(g.winner * player), p=p)

    return [(g.winner, len(g.players)) for g in games]


def self_play_episode(replay_buffer):
    """Single-game self-play episode (tests / debugging).
    Returns (winner, number of moves)."""
    return self_play_episodes(replay_buffer, n_games=1)[0]


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


def _new_histories():
    return {
        'outcomes': [],       # winner of each self-play game (+1 / -1 / 0)
        'game_lengths': [],   # plies per self-play game
        'p1_win_rates': [],   # trailing share of games won by player +1
        'draw_rates': [],     # trailing share of drawn (truncated) games
        'v_losses': [],
        'p_losses': [],
        'eval_history': [],   # (episode, results dict)
    }


def _write_plots(h):
    """Render the metric plots into CHECKPOINT_DIR (refreshed during the
    run so long Colab sessions can be monitored from Drive)."""
    game_lengths = h['game_lengths']

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
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'game_length.png'))

    plt.figure()
    plt.plot(h['p1_win_rates'], label='P1 win rate')
    plt.plot(h['draw_rates'], label='draw rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f'self-play outcomes (trailing {METRIC_WINDOW} episodes)')
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'win_rate.png'))

    plt.figure()
    plt.plot(h['v_losses'])
    plt.title('value loss')
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'value_loss.png'))

    plt.figure()
    plt.plot(h['p_losses'])
    plt.title('policy loss')
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'policy_loss.png'))

    plt.close('all')


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
    histories = _new_histories()
    start_episode = 0

    state = load_checkpoint(CHECKPOINT_DIR, CHECKPOINT_FILE)
    if state is not None:
        if state['meta'] != CHECKPOINT_META:
            raise SystemExit(
                f'checkpoint in {CHECKPOINT_DIR} is for {state["meta"]}, '
                f'but the current config is {CHECKPOINT_META}; move the '
                f'old checkpoint away or point AZ_CHECKPOINT_DIR elsewhere')
        restore_networks(state, policy_v, policy_p, GAME_OBS)
        restore_buffer(state, replay_buffer)
        restore_rng(state)
        histories = state['histories']
        start_episode = state['episode']
        print(f'resumed from checkpoint: episode {start_episode}, '
              f'{len(replay_buffer)} buffered positions', flush=True)
    else:
        print(f'no checkpoint in {CHECKPOINT_DIR}; starting fresh',
              flush=True)

    outcomes = histories['outcomes']
    game_lengths = histories['game_lengths']

    e = start_episode
    while e < EPISODES:
        # One parallel batch of self-play games counts as that many episodes;
        # per-episode metrics / training / eval / checkpointing below keep
        # the exact cadence of the old one-game-at-a-time loop.
        batch_results = self_play_episodes(
            replay_buffer, n_games=min(PARALLEL_GAMES, EPISODES - e))

        for winner, moves in batch_results:
            outcomes.append(winner)
            game_lengths.append(moves)

            window = outcomes[-METRIC_WINDOW:]
            histories['p1_win_rates'].append(window.count(1) / len(window))
            histories['draw_rates'].append(window.count(0) / len(window))

            print(f'episode {e + 1}: winner {winner:+d} in {moves} moves, '
                  f'last {len(window)}: '
                  f'P1 {histories["p1_win_rates"][-1]:.2f} / '
                  f'draw {histories["draw_rates"][-1]:.2f}, '
                  f'avg length {np.mean(game_lengths[-METRIC_WINDOW:]):.1f}',
                  flush=True)

            if len(replay_buffer) > BATCH_SIZE:
                for _ in range(TRAIN_BATCHES_PER_EPISODE):
                    loss_v, loss_p = train_networks(replay_buffer)
                histories['v_losses'].append(loss_v)
                histories['p_losses'].append(loss_p)

            if (e + 1) % EVAL_EVERY == 0:
                results = evaluate_vs_random()
                histories['eval_history'].append((e + 1, results))
                print(f'--- eval vs random after episode {e + 1}: {results}',
                      flush=True)
                _write_plots(histories)

            e += 1

        # Checkpoint at most once per parallel batch (episodes inside a
        # batch finish together, so mid-batch checkpoints could not save
        # any resume time): save whenever the batch crossed a
        # CHECKPOINT_EVERY boundary, and always at the end of the run.
        crossed = (e // CHECKPOINT_EVERY
                   != (e - len(batch_results)) // CHECKPOINT_EVERY)
        if crossed or e == EPISODES:
            save_checkpoint(CHECKPOINT_DIR, CHECKPOINT_FILE, e,
                            policy_v, policy_p, replay_buffer,
                            histories, CHECKPOINT_META)

    # persist the trained networks on their own (weights-only files usable
    # without the training state, namespaced by game)
    policy_v.save_weights(
        os.path.join(CHECKPOINT_DIR, f'{GAME_NAME}_policy_v.weights.h5'))
    policy_p.save_weights(
        os.path.join(CHECKPOINT_DIR, f'{GAME_NAME}_policy_p.weights.h5'))

    _write_plots(histories)

    if histories['eval_history']:
        print('\neval-vs-random history:')
        for episode, results in histories['eval_history']:
            print(f'  episode {episode}: {results}')


if __name__ == '__main__':
    train()
