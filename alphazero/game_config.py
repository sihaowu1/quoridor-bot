"""
Central place for the game configuration shared by the NN, the MCTS engine
and the training driver.

The game must expose the two-player conventions documented in tic_tac_toe.py
and quoridor.py: gymnasium-style ``reset``/``step``, canonical observations
(always from the perspective of the player to move), rewards from the
mover's perspective, ``legal_actions()``, ``to_play`` and ``info['winner']``
at the end of a game.

The default game is Quoridor.  Set the AZ_GAME environment variable to
``ttt`` to switch the whole pipeline (networks, MCTS, training, tests) back
to tic-tac-toe, e.g. to re-run the original validation suite:

    AZ_GAME=ttt python -m alphazero.test_alphazero_ttt
"""

import os

GAME = os.environ.get('AZ_GAME', 'quoridor').lower()

if GAME in ('ttt', 'tictactoe', 'tic_tac_toe'):
    from alphazero.tic_tac_toe import TicTacToe, GAME_ACTIONS, GAME_OBS

    GAME_NAME = 'TicTacToe'

    def make_game():
        return TicTacToe()

else:
    from alphazero.quoridor import Quoridor, num_actions, obs_size

    GAME_NAME = 'Quoridor'

    # The one place to change when scaling up: the full game is
    # BOARD_SIZE = 9, WALLS = 10 (expect training to need far more
    # compute; see README).
    BOARD_SIZE = 9
    WALLS = 10

    GAME_ACTIONS = num_actions(BOARD_SIZE)
    GAME_OBS = obs_size(BOARD_SIZE)

    def make_game():
        return Quoridor(BOARD_SIZE, WALLS)
