import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        reward = rate > np.random.rand()
        return int(reward)


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        a, r = action, reward
        self.ns[a] += 1
        self.qs[a] += (r - self.qs[a]) / self.ns[a]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.qs))
        return np.argmax(self.qs)


if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    sum_r = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        sum_r += reward

        total_rewards.append(sum_r)
        rates.append(sum_r / (step+1))

    print(sum_r)

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()
