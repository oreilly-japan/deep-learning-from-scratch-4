import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from common.gridworld import GridWorld


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)  # hidden_size
        self.l2 = L.Linear(4)  # action_size

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state_vec, action, reward, next_state_vec, done):
        if done:
            next_q = np.zeros(1)  # [0.]
        else:
            next_qs = self.qnet(next_state_vec)
            next_q = next_qs.max(axis=1)
            next_q.unchain()

        target = self.gamma * next_q + reward
        qs = self.qnet(state_vec)
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data


env = GridWorld()
agent = QLearningAgent()

episodes = 1000
loss_list = []

for episode in range(episodes):
    state = env.reset()
    state_vec = one_hot(state)
    avg_loss = []
    done = False

    while not done:
        action = agent.get_action(state_vec)
        next_state, reward, done = env.step(action)

        next_state_vec = one_hot(next_state)
        loss = agent.update(state_vec, action, reward, next_state_vec, done)
        avg_loss.append(loss)

        state_vec = next_state_vec

    loss_list.append(np.average(avg_loss))


plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_list)), loss_list)
plt.show()

# visualize
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)

#env.render_v(None, agent.policy)