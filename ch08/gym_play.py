import numpy as np
import gym


env = gym.make('CartPole-v0')
state = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, info = env.step(action)
env.close()