from collections import namedtuple, deque
import random

"""
ReplayBuffer class stores game plays that are used for nn training
Specifically, it stores:
* State
* target value
* observation (i.e. state) at previous step
* target policy (according to visit counts) 
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
                                     field_names=['obs', 'v', 'p_obs', 'p']) 
    

    def add(self, obs, v, p, p_obs): 
        e = self.experience(obs, v, p, p_obs) 
        self.memory.append(e) 
    

    def sample(self): 
        experiences = random.sample(self.memory, k=self.batch_size) 
        return experiences
    

    def __len__(self): 
        return len(self.memory) 