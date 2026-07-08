import tensorflow as tf
from tensorflow import keras
from alphazero.game_config import GAME_ACTIONS, GAME_NAME

tf.keras.backend.set_floatx('float32')

"""
One dual-head network, as in the AlphaZero paper: a shared trunk feeds

* a value head estimating the game outcome in [-1, 1] for the player to
  move, and
* a policy head estimating the MCTS visit-count policy over the full
  action space.

Sharing one trunk between the heads (the paper's "dual" architecture,
replacing the earlier two separate networks) means both objectives train
the same features — the value head, which only gets one scalar target per
position, is regularised by the much denser policy signal — and every MCTS
leaf evaluation runs the trunk once instead of twice.

Two trunks share the AlphaZeroNet name:

* Quoridor: the paper's residual conv tower at hobby scale, over
  cell-aligned 2D planes.  The flat canonical observation is decoded
  inside call(), so the rest of the pipeline (MCTS, replay buffer,
  checkpoints) keeps passing flat vectors.
* TicTacToe (AZ_GAME=ttt): a small dense MLP on the raw 9-vector.

Training runs the paper's joint objective (see train_step):

    loss = (z - v)^2 + CrossEntropy(pi, p) + c * ||theta||^2

ARCH names the architecture and is stamped into the checkpoint metadata by
run.py, so weights from one architecture are never loaded into another.
"""

L2_REG = 1e-4  # the paper's weight-decay coefficient c


class _AlphaZeroBase(keras.Model):
    """Shared dual-head plumbing: subclasses build a trunk plus the two
    heads and implement ``call(obs, training) -> (v, p)``; the joint
    training step lives here."""

    def __init__(self):
        # Explicit name: an auto-generated name starting with '_' is not a
        # valid TF root name scope, so any graph-mode call (tf.function
        # tracing) would fail.
        super().__init__(name='alphazero_net')
        self._loss_v = keras.losses.MeanSquaredError()
        self._loss_p = keras.losses.CategoricalCrossentropy()

    def train_step(self, data):
        """One gradient step on the joint AlphaZero loss.

        The L2 term is collected from the layers' kernel_regularizers via
        ``self.losses``.  Explicit train_step (rather than compiled
        multi-output losses) so the value/policy split is reported under
        stable names — run.py reads ``loss_v`` / ``loss_p`` from the logs.
        """
        x, y = data[0], data[1]
        v_target, p_target = y

        with tf.GradientTape() as tape:
            v, p = self(x, training=True)
            loss_v = self._loss_v(v_target, v)
            loss_p = self._loss_p(p_target, p)
            reg = tf.add_n(self.losses) if self.losses else tf.constant(0.0)
            loss = loss_v + loss_p + reg

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(grads, self.trainable_variables)

        return {'loss': loss, 'loss_v': loss_v, 'loss_p': loss_p}


HIDDEN_STATES = 128


class _DenseNet(_AlphaZeroBase):
    """Dense trunk + the two heads, for games with tiny flat observations
    (tic-tac-toe).  Successor of the original pair of 2x128 MLPs."""

    def __init__(self):
        super().__init__()

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

        self.p_out = keras.layers.Dense(GAME_ACTIONS,
                                        activation='softmax',
                                        kernel_initializer=keras.initializers.he_normal(),
                                        name='p-out')

    def call(self, input, training=None):
        x = self.dense1(input)
        x = self.dense2(x)

        return self.v_out(x), self.p_out(x)


if GAME_NAME == 'Quoridor':
    from alphazero.game_config import BOARD_SIZE

    # The paper's trunk is 256 filters x 19 (or 39) residual blocks — sized
    # for thousands of TPUs generating self-play.  At 64 x 6 (~0.45M
    # parameters) the tower keeps the paper's shape but fits a single-GPU
    # Colab budget; 13 successive 3x3 convs give a 27x27 receptive field,
    # so every output cell sees the whole 9x9 board (the previous plain
    # 3-conv trunk saw only 7x7 — it literally could not connect a wall on
    # one side of the board to a pawn on the other).
    FILTERS = 64
    BLOCKS = 6

    ARCH = f'resnet-{FILTERS}x{BLOCKS}'

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

    def _conv(filters, size, name):
        # use_bias=False: every conv here is followed by BatchNorm, whose
        # beta makes a conv bias redundant.
        return keras.layers.Conv2D(filters, size,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=keras.initializers.he_normal(),
                                   kernel_regularizer=keras.regularizers.l2(L2_REG),
                                   name=name)

    def _dense(units, activation, name):
        return keras.layers.Dense(units,
                                  activation=activation,
                                  kernel_initializer=keras.initializers.he_normal(),
                                  kernel_regularizer=keras.regularizers.l2(L2_REG),
                                  name=name)

    class _ResidualBlock(keras.layers.Layer):
        """conv-BN-ReLU-conv-BN, skip connection, ReLU (the paper's
        block)."""

        def __init__(self, index):
            super().__init__(name=f'res-{index}')
            self.conv1 = _conv(FILTERS, 3, 'conv-1')
            self.bn1 = keras.layers.BatchNormalization(name='bn-1')
            self.conv2 = _conv(FILTERS, 3, 'conv-2')
            self.bn2 = keras.layers.BatchNormalization(name='bn-2')

        def call(self, x, training=None):
            y = tf.nn.relu(self.bn1(self.conv1(x), training=training))
            y = self.bn2(self.conv2(y), training=training)
            return tf.nn.relu(x + y)

    class _ConvNet(_AlphaZeroBase):
        """The paper's network: conv stem, residual tower, then

        * policy head: 1x1 conv (2 filters) - BN - ReLU - dense softmax
          over the full flat action space (pawn moves + wall slots);
        * value head: 1x1 conv (1 filter) - BN - ReLU - dense 64 - ReLU -
          dense 1, tanh.
        """

        def __init__(self):
            super().__init__()

            self.stem = _conv(FILTERS, 3, 'stem')
            self.stem_bn = keras.layers.BatchNormalization(name='stem-bn')
            self.blocks = [_ResidualBlock(i + 1) for i in range(BLOCKS)]

            self.p_conv = _conv(2, 1, 'p-conv')
            self.p_bn = keras.layers.BatchNormalization(name='p-bn')
            self.p_flatten = keras.layers.Flatten(name='p-flatten')
            self.p_out = _dense(GAME_ACTIONS, 'softmax', 'p-out')

            self.v_conv = _conv(1, 1, 'v-conv')
            self.v_bn = keras.layers.BatchNormalization(name='v-bn')
            self.v_flatten = keras.layers.Flatten(name='v-flatten')
            self.v_dense = _dense(64, 'relu', 'v-dense')
            # tanh bounds the value to [-1, 1] (see _DenseNet.v_out).
            self.v_out = _dense(1, 'tanh', 'v_out')

        def call(self, input, training=None):
            # training is threaded through explicitly so BatchNorm uses
            # batch statistics in train_step and its moving statistics
            # everywhere else (mcts.predict_fn passes training=False).
            x = _quoridor_planes(input)
            x = tf.nn.relu(self.stem_bn(self.stem(x), training=training))
            for block in self.blocks:
                x = block(x, training=training)

            v = tf.nn.relu(self.v_bn(self.v_conv(x), training=training))
            v = self.v_out(self.v_dense(self.v_flatten(v)))

            p = tf.nn.relu(self.p_bn(self.p_conv(x), training=training))
            p = self.p_out(self.p_flatten(p))

            return v, p

    AlphaZeroNet = _ConvNet

else:
    ARCH = f'dense-{HIDDEN_STATES}x2'
    AlphaZeroNet = _DenseNet
