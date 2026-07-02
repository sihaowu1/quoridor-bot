"""
Central place for the game configuration shared by the NN, the MCTS engine
and the training driver.

The game must expose the two-player conventions documented in tic_tac_toe.py:
gymnasium-style ``reset``/``step``, canonical observations (always from the
perspective of the player to move), rewards from the mover's perspective,
``legal_actions()``, ``to_play`` and ``info['winner']`` at the end of a game.
Quoridor will plug in here later by satisfying the same interface.
"""

from alphazero.tic_tac_toe import TicTacToe, GAME_ACTIONS, GAME_OBS

GAME_NAME = 'TicTacToe'


def make_game():
    return TicTacToe()
