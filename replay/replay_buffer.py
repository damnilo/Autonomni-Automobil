import random

from collections import deque
from replay.transition import Transition

class ReplayBuffer:

    def __init__(self, capacity):

        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):

        self.buffer.append(transition)

    def sample(self, batch_size):

        return random.sample(self.buffer, batch_size)
    
    def __len__(self):

        return len(self.buffer)