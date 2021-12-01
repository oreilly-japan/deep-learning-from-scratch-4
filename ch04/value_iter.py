if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import get_greedy_policy


def value_iter_onestep(env, gamma, V):
    delta = 0

    for state in env.states():
        action_values = []

        for action in env.actions():
            next_state = env.next_state(state, action)

            if next_state is not None:
                r = env.reward(state, action, next_state)
                value = r + gamma * V[next_state]
                action_values.append(value)

        if len(action_values) > 0:
            new_value = max(action_values)
            delta = max(delta, abs(new_value - V[state]))
            V[state] = new_value

    return V, delta


def value_iter(env, gamma, threshold=0.001, is_render=True):
    V = defaultdict(lambda: 0)

    while True:
        if is_render:
            env.render_v(V)

        V, delta = value_iter_onestep(env, gamma, V)
        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    V = value_iter(env, gamma)

    pi = get_greedy_policy(V, env, gamma)
    env.render_v(V, pi)
