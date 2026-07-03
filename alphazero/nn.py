import tensorflow as tf
from tensorflow import keras
from alphazero.game_config import GAME_ACTIONS, GAME_NAME

tf.keras.backend.set_floatx('float32')

"""
2 neural networks
* 1 for value estimation
* 1 for policy estimation

Two architectures share the PolicyV / PolicyP names:

* Quoridor: a small convolutional net over cell-aligned 2D planes.  The
  flat canonical observation is decoded inside call(), so the rest of the
  pipeline (MCTS, replay buffer, checkpoints) keeps passing flat vectors.
* TicTacToe (AZ_GAME=ttt): the original dense MLP on the raw 9-vector.
"""

HIDDEN_STATES = 128


class _DensePolicyV(keras.Model):
    """
    This is the value neural network, which given a state, approximates the value of the node
    """

    def __init__(self):

        super(_DensePolicyV, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense-1')

        self.dense2 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense-2')

        # tanh bounds the value to [-1, 1]: a game outcome can never beat a
        # win (+1), so an unbounded head lets an untrained/noisy network
        # score arbitrary moves above proven wins in the PUCT selection.
        self.v_out = keras.layers.Dense(1,
                                        activation='tanh',
                                        kernel_initializer=keras.initializers.he_normal(),
                                        name='v_out')

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.v_out(x)

        return x


class _DensePolicyP(keras.Model):
    """
    The Policy neural network estimates the MCTS policy for some nodes given a state
    """

    def __init__(self):

        super(_DensePolicyP, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense-1')

        self.dense2 = keras.layers.Dense(HIDDEN_STATES,
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense-2')

        self.p_out = keras.layers.Dense(GAME_ACTIONS,
                                        activation='softmax',
                                        kernel_initializer=keras.initializers.he_normal(),
                                        name='p-out')

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.p_out(x)

        return x


if GAME_NAME == 'Quoridor':
    from alphazero.game_config import BOARD_SIZE

    FILTERS = 64
    TRUNK_CONVS = 3

    def _quoridor_planes(x):
        """Decode the flat canonical observation into 8 cell-aligned
        (n, n) planes:

            0  own pawn one-hot
            1  opponent pawn one-hot
            2  N-blocked: 1 where the step from this cell toward row 0
               is blocked by a wall or the board edge
            3  S-blocked, 4 E-blocked, 5 W-blocked (same convention)
            6  own walls-left fraction (constant plane)
            7  opponent walls-left fraction (constant plane)

        Walls are given to the network as the moves they forbid (the same
        derivation as quoridor._step_blocked, with board edges baked in as
        blocked) instead of raw (n-1, n-1) slot grids, so the first conv
        layer sees exact cell-aligned geometry rather than having to learn
        the half-cell slot offset.
        """
        n, m = BOARD_SIZE, BOARD_SIZE - 1
        own = tf.reshape(x[:, :n * n], (-1, n, n))
        opp = tf.reshape(x[:, n * n:2 * n * n], (-1, n, n))
        base = 2 * n * n
        hw = tf.reshape(x[:, base:base + m * m], (-1, m, m))
        vw = tf.reshape(x[:, base + m * m:base + 2 * m * m], (-1, m, m))

        # Boundary occupancy: bh[r, c] = 1 iff a wall crosses the boundary
        # between rows r and r+1 at column c, i.e. hw[r, c-1] or hw[r, c]
        # (out-of-range slots read as empty).  bv likewise for columns.
        bh = tf.maximum(tf.pad(hw, [[0, 0], [0, 0], [1, 0]]),
                        tf.pad(hw, [[0, 0], [0, 0], [0, 1]]))  # (B, m, n)
        bv = tf.maximum(tf.pad(vw, [[0, 0], [1, 0], [0, 0]]),
                        tf.pad(vw, [[0, 0], [0, 1], [0, 0]]))  # (B, n, m)

        # Shift the boundaries onto the cells on each side; the padded
        # row/column of ones is the board edge, which also blocks.
        blocked_n = tf.pad(bh, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        blocked_s = tf.pad(bh, [[0, 0], [0, 1], [0, 0]], constant_values=1)
        blocked_e = tf.pad(bv, [[0, 0], [0, 0], [0, 1]], constant_values=1)
        blocked_w = tf.pad(bv, [[0, 0], [0, 0], [1, 0]], constant_values=1)

        planes = tf.stack([own, opp,
                           blocked_n, blocked_s, blocked_e, blocked_w],
                          axis=-1)                              # (B, n, n, 6)
        walls = tf.tile(tf.reshape(x[:, -2:], (-1, 1, 1, 2)), (1, n, n, 1))
        return tf.concat([planes, walls], axis=-1)              # (B, n, n, 8)

    def _trunk():
        return [keras.layers.Conv2D(FILTERS, 3,
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=keras.initializers.he_normal(),
                                    name=f'conv-{i + 1}')
                for i in range(TRUNK_CONVS)]

    class _ConvPolicyV(keras.Model):
        """Value network: conv trunk over the planes, 1x1 head, tanh."""

        def __init__(self):
            super(_ConvPolicyV, self).__init__()
            self.convs = _trunk()
            self.head = keras.layers.Conv2D(2, 1,
                                            activation='relu',
                                            kernel_initializer=keras.initializers.he_normal(),
                                            name='v-head')
            self.flatten = keras.layers.Flatten(name='v-flatten')
            self.dense = keras.layers.Dense(64,
                                            activation='relu',
                                            kernel_initializer=keras.initializers.he_normal(),
                                            name='v-dense')
            # tanh bounds the value to [-1, 1] (see _DensePolicyV.v_out).
            self.v_out = keras.layers.Dense(1,
                                            activation='tanh',
                                            kernel_initializer=keras.initializers.he_normal(),
                                            name='v_out')

        def call(self, input):
            x = _quoridor_planes(input)
            for conv in self.convs:
                x = conv(x)
            x = self.head(x)
            x = self.flatten(x)
            x = self.dense(x)

            return self.v_out(x)

    class _ConvPolicyP(keras.Model):
        """Policy network: conv trunk over the planes, 1x1 head, softmax
        over the full flat action space (pawn moves + wall slots)."""

        def __init__(self):
            super(_ConvPolicyP, self).__init__()
            self.convs = _trunk()
            self.head = keras.layers.Conv2D(4, 1,
                                            activation='relu',
                                            kernel_initializer=keras.initializers.he_normal(),
                                            name='p-head')
            self.flatten = keras.layers.Flatten(name='p-flatten')
            self.p_out = keras.layers.Dense(GAME_ACTIONS,
                                            activation='softmax',
                                            kernel_initializer=keras.initializers.he_normal(),
                                            name='p-out')

        def call(self, input):
            x = _quoridor_planes(input)
            for conv in self.convs:
                x = conv(x)
            x = self.head(x)
            x = self.flatten(x)

            return self.p_out(x)

    PolicyV = _ConvPolicyV
    PolicyP = _ConvPolicyP

else:
    PolicyV = _DensePolicyV
    PolicyP = _DensePolicyP
