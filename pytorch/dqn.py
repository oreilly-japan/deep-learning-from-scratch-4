import copy
from collections import deque
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import RecordVideo
import os


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

        state = torch.tensor(np.array([x[0] for x in data]), dtype=torch.float32)
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))#action = torch.tensor(np.array([x[1] for x in data]).astype(np.long))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.array([x[3] for x in data]), dtype=torch.float32)
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


episodes = 300
sync_interval = 20
render_interval = 50  # 每50个episode渲染一次
video_dir = "./dqn_videos"  # 更改视频保存目录
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

print(f"视频将保存到: {os.path.abspath(video_dir)}")

# 使用RecordVideo包装环境，指定要记录的episodes
base_env = gym.make('CartPole-v0', render_mode="rgb_array")
env = RecordVideo(
    base_env, 
    video_dir,
    episode_trigger=lambda x: x % render_interval == 0,  # 每render_interval个episode录制一次
    name_prefix="cartpole-dqn"  # 添加前缀以区分不同的运行
)

agent = DQNAgent()
reward_history = []

for episode in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):  # Handle new gym API
        state, _ = state
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        step_result = env.step(action)
        
        # Handle different gym versions
        if len(step_result) == 5:  # Newer gym version (step returns 5 values)
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # Older gym version (step returns 4 values)
            next_state, reward, done, info = step_result

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))

# 添加环境关闭
env.close()
print(f"训练完成！视频已保存到 {os.path.abspath(video_dir)} 目录")