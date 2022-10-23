import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld

env = GridWorld()

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4
        self.xi = 0.95 #エリジビリティ減衰率、lambdaは既に用いられているのでxiを使った(なので実際はTD(xi)法)
        self.z = defaultdict(lambda:0) # エリジビリティ・トレース
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25} #行動の確率分布
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state, reward, next_state, done):
      next_V = 0 if done else self.V[next_state]
      target = reward + self.gamma * next_V  
      for h in range(env.height):
          for w in range(env.width):
            #エリジビリティトレースを更新
            if state == (h, w):
                self.z[state] = 1 + self.gamma*self.xi*self.z[state]
            else:
                self.z[state] =self.gamma*self.xi*self.z[state]       
            self.V[state] += (target - self.V[state]) * self.alpha * self.z[state]

agent = TdAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.eval(state, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_v(agent.V)