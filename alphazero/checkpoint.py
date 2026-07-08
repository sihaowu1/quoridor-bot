"""
Crash-safe training checkpoints for the AlphaZero driver.

Everything needed to resume a run is captured in ONE pickle file:
  * the network's weights and its Adam optimizer state,
  * the full replay buffer,
  * the training metric histories (outcomes, lengths, losses, evals),
  * the Python / numpy RNG states,
  * a meta block (game name, action/observation sizes) that the loader
    validates so a checkpoint is never applied to a mismatched config.

Saves are atomic: the state is written to a temp file and swapped in with
``os.replace``, and the previous checkpoint is kept as ``<file>.bak``.  A
runtime dying mid-write (the Colab failure mode this exists for) therefore
never destroys the last good state; the loader falls back to the ``.bak``
if the primary file is unreadable.
"""

import os
import pickle
import random

import numpy as np


def _optimizer_state(model):
    opt = model.optimizer
    if opt is None or not getattr(opt, 'built', False):
        return None  # no train step has run yet; nothing to save
    return [np.asarray(v) for v in opt.variables]


def _restore_optimizer(model, state):
    if state is None:
        return
    opt = model.optimizer
    if not getattr(opt, 'built', False):
        opt.build(model.trainable_variables)
    if len(opt.variables) != len(state):
        raise ValueError(
            f'optimizer state mismatch: checkpoint has {len(state)} '
            f'variables, model expects {len(opt.variables)}')
    for var, val in zip(opt.variables, state):
        var.assign(val)


def save_checkpoint(directory, filename, episode, network,
                    replay_buffer, histories, meta):
    os.makedirs(directory, exist_ok=True)
    state = {
        'meta': dict(meta),
        'episode': episode,
        'weights': network.get_weights(),
        'optimizer': _optimizer_state(network),
        'buffer': [(np.asarray(e.obs), float(e.v), np.asarray(e.p))
                   for e in replay_buffer.memory],
        'histories': histories,
        'py_random': random.getstate(),
        'np_random': np.random.get_state(),
    }

    final = os.path.join(directory, filename)
    tmp = final + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    if os.path.exists(final):
        os.replace(final, final + '.bak')
    os.replace(tmp, final)


def load_checkpoint(directory, filename):
    """Return the newest readable checkpoint state, or None if there is
    no (readable) checkpoint yet."""
    final = os.path.join(directory, filename)
    for path in (final, final + '.bak'):
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as err:  # corrupt file: fall through to the .bak
            print(f'warning: could not read checkpoint {path}: {err}')
    return None


def restore_network(state, network, obs_dim):
    if 'weights' not in state:
        # 'v_weights'/'p_weights' era: two separate networks, incompatible
        # with the dual-head architecture (run.py's meta check normally
        # catches this first via the 'arch' key; this guards direct loads,
        # e.g. play/agent.py with AZ_CHECKPOINT).
        raise ValueError(
            'checkpoint predates the dual-head network and cannot be '
            'loaded; retrain, or check out the two-network revision')
    # Subclassed keras models must be built (called once) before weights
    # can be assigned.
    dummy = np.zeros((1, obs_dim), dtype=np.float32)
    network(dummy)
    network.set_weights(state['weights'])
    _restore_optimizer(network, state['optimizer'])


def restore_buffer(state, replay_buffer):
    for obs, v, p in state['buffer']:
        replay_buffer.add(obs=obs, v=v, p=p)


def restore_rng(state):
    random.setstate(state['py_random'])
    np.random.set_state(state['np_random'])
