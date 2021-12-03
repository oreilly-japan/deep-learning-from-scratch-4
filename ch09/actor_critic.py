import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from common.utils import plot_total_reward


class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(256)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x


class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(256)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0001
        self.lr_v = 0.0001
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()

        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        state = np.atleast_2d(state) #state[np.newaxis, :]  # add batch axis
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice([0, 1], p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = np.atleast_2d(state)  # add batch axis
        next_state = np.atleast_2d(next_state)

        td_target = reward + self.gamma * self.v(next_state) * (1 - done)
        td_target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, td_target)

        delta = td_target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


env = gym.make('CartPole-v0')
agent = Agent()
reward_log = {}

for episode in range(3000):
    state = env.reset()
    done = False
    sum_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        sum_reward += reward

    reward_log[episode] = sum_reward
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, sum_reward))

plot_total_reward(reward_log)