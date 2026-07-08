"""
Minimal HTTP server for playing against the trained bot in a browser.

Stdlib-only (http.server), single global game, human is always player +1
(bottom pawn, moves first, aims for the top row).  Because the canonical
frame of player +1 IS the absolute frame, everything the browser sees --
pawn destinations, wall slots, action indices -- needs no translation.

Run from the repo root:

    AZ_BACKEND=py uv run python -m play.server

then open http://localhost:8000.  Environment knobs:

    AZ_PLAY_SIMS   MCTS simulations per bot move (default 200)
    AZ_PLAY_PORT   port (default 8000)
    AZ_CHECKPOINT / AZ_WEIGHTS   see play/agent.py
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from alphazero import mcts
from alphazero.game_config import make_game
from alphazero.quoridor import decode_action
from play.agent import bot_move, load_network

HUMAN, BOT = 1, -1

# The bot is only trained for 3 walls per player, so pin play to 3
# regardless of game_config.WALLS (which may be raised for training a
# full 10-wall game).  Pinning the game itself — not just the displayed
# counter — keeps the legal wall moves at 3 too.
PLAY_WALLS = 3

_HERE = os.path.dirname(os.path.abspath(__file__))


class Session:
    """One game plus the bits of per-move context the UI wants."""

    def __init__(self):
        template = make_game()  # carries the configured backend / board size
        self.game = type(template)(template.n, PLAY_WALLS,
                                   max_moves=template.max_moves)
        self.reset()

    def reset(self):
        self.obs, _ = self.game.reset()
        self.value = None       # bot's root value estimate, bot perspective
        self.last_bot = None    # human-readable last bot move

    def play_human(self, action):
        game = self.game
        if game.done or game.to_play != HUMAN:
            raise ValueError('not your turn')
        if action not in game.legal_actions():
            raise ValueError(f'illegal action {action}')
        self.obs, _, _, _, _ = game.step(action)

        if not game.done:
            self._play_bot()

    def _play_bot(self):
        game = self.game
        action, self.value = bot_move(game, self.obs)
        # Decode for display, flipping the bot's canonical frame back to
        # absolute coordinates (rows are mirrored for player -1).
        decoded = decode_action(action, game.n)
        self.obs, _, _, _, _ = game.step(action)
        if decoded[0] == 'move':
            r, c = game.pos[BOT]
            self.last_bot = f'pawn to ({r}, {c})'
        else:
            o, r, c = decoded
            self.last_bot = f'{o} wall at ({game.n - 2 - r}, {c})'

    def state(self):
        game = self.game
        state = {
            'n': game.n,
            'pos': {'human': list(game.pos[HUMAN]),
                    'bot': list(game.pos[BOT])},
            'walls_left': {'human': game.walls_left[HUMAN],
                           'bot': game.walls_left[BOT]},
            'h_walls': sorted(list(w) for w in game.h_walls),
            'v_walls': sorted(list(w) for w in game.v_walls),
            'done': game.done,
            'winner': ({1: 'human', -1: 'bot', 0: 'draw'}[game.winner]
                       if game.done else None),
            # bot's value negated = position from the human's perspective
            'value': -self.value if self.value is not None else None,
            'last_bot': self.last_bot,
            'legal': [],
        }
        if not game.done and game.to_play == HUMAN:
            hr, hc = game.pos[HUMAN]
            for action in game.legal_actions():
                decoded = decode_action(action, game.n)
                if decoded[0] == 'move':
                    dr, dc = decoded[1]
                    state['legal'].append({'action': action, 'kind': 'move',
                                           'dest': [hr + dr, hc + dc]})
                else:
                    o, r, c = decoded
                    state['legal'].append({'action': action, 'kind': 'wall',
                                           'o': o, 'r': r, 'c': c})
        return state


session = Session()


class Handler(BaseHTTPRequestHandler):

    def _send(self, code, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else \
            json.dumps(body).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(_HERE, 'index.html'), 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        try:
            if self.path == '/new':
                session.reset()
            elif self.path == '/move':
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length) or b'{}')
                session.play_human(int(body['action']))
            else:
                self._send(404, {'error': 'not found'})
                return
        except (ValueError, KeyError) as err:
            self._send(400, {'error': str(err)})
            return
        self._send(200, session.state())

    def log_message(self, fmt, *args):  # quieter default logging
        if not (args and str(args[0]).startswith('GET')):
            super().log_message(fmt, *args)


def main():
    mcts.MCTS_POLICY_EXPLORE = int(os.environ.get('AZ_PLAY_SIMS', '200'))
    print('loading network:', load_network())
    print(f'{mcts.MCTS_POLICY_EXPLORE} simulations per bot move')

    port = int(os.environ.get('AZ_PLAY_PORT', '8000'))
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'play at http://localhost:{port}')
    server.serve_forever()


if __name__ == '__main__':
    main()
