from collections import namedtuple, deque
import random

"""
ReplayBuffer class stores self-play positions used for nn training.
Each experience is one position:
* obs: canonical observation of the position (perspective of the player to move)
* v: target value — the final game outcome from that same player's
     perspective (+1 win, 0 draw, -1 loss)
* p: target policy — the MCTS visit-count distribution over the full action
     space computed at that position
"""

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """
        Params:
        * buffer_size (int): maximum size of buffer
        * batch_size (int): size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=['obs', 'v', 'p'])


    def add(self, obs, v, p):
        # Build by keyword so callers can't silently transpose fields.
        e = self.experience(obs=obs, v=v, p=p)
        self.memory.append(e)


    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        return experiences


    def __len__(self):
        return len(self.memory)
