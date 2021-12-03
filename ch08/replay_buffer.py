from collections import deque
import random
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int)
        return state, action, reward, next_state, done


env = gym.make('CartPole-v1')
replay_buffer = ReplayBuffer(buffer_size=100, batch_size=3)
state = env.reset()
done = False

while not done:
    action = 0
    next_state, reward, done, info = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)

state, action, reward, next_state, done = replay_buffer.get_batch()
print(state.shape)  # (3, 4)
print(action.shape)  # (3,)
print(reward.shape)  # (3,)
print(next_state.shape)  # (3, 4)
print(done.shape)  # (3,)