import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_action_probs
from ch05.mc_control import McAgent


class McOffPolicyAgent(McAgent):
    def __init__(self):
        super().__init__()
        self.b = self.pi.copy()

    def get_action(self, state):
        ps = self.b[state]
        actions, probs = list(ps.keys()), list(ps.values())
        return np.random.choice(actions, p=probs)

    def update(self):
        g = 0
        rho = 1

        for data in reversed(self.experience):
            state, action, reward = data
            key = (state, action)

            g = self.gamma * rho * g + reward
            self.Q[key] += (g - self.Q[key]) * self.alpha
            rho *= self.pi[state][action] / self.b[state][action]

            self.pi[state] = greedy_action_probs(self.Q, state, epsilon=0)
            self.b[state] = greedy_action_probs(self.Q, state, self.epsilon)


if __name__ == '__main__':
    env = GridWorld()
    agent = McOffPolicyAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update()
                break

            state = next_state

    env.render_q(agent.Q)