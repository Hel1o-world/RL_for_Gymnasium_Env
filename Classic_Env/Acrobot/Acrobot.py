import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import deque
import random
import os


class Qnet(nn.Module): # 输入为观测，输出为动作值
    def __init__(self, obs_dim, act_dim):
        super(Qnet, self).__init__()
        self.l1 = nn.Linear(obs_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, act_dim)
        self.bn1 = nn.LayerNorm(128)
        self.bn2 = nn.LayerNorm(64)
        self.dr1 = nn.Dropout(0.3)
        self.dr2 = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.bn1(x)
        x = self.dr1(x)
        x = F.relu(self.l2(x))
        x = self.bn2(x)
        x = self.dr2(x)
        return self.l3(x)

class Acrobot:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 device=torch.device('cpu'),
                 epsilon=0.1,
                 epsilon_end = 0.01,
                 gamma=0.9,
                 lr=1e-3,
                 ):
        self.device = device
        self.epsilon_end = epsilon_end
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.qnet = Qnet(self.obs_dim, self.act_dim).to(self.device)
        self.qnet_target = Qnet(self.obs_dim, self.act_dim).to(self.device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.optimizer = optim.Adam(self.qnet.parameters(), self.lr)
        self.loss = nn.MSELoss()
        self.buffer_size = 10000
        self.batch_size = 512
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size,device=self.device)


    def sync_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, obs):
        self.epsilon = max(self.epsilon_end, self.epsilon - 0.00002)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
            with torch.no_grad():
                q_values = self.qnet(obs)
            return torch.argmax(q_values).item() #item()返回的时python值，在cpu上

    def update(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
        if len(self.buffer) < self.batch_size:
            return

        obs, action, reward, next_obs, done = self.buffer.get_batch()

        q_values = self.qnet(obs)
        q_value = q_values[torch.arange(self.batch_size),action]
        with torch.no_grad():
            q_values_next = self.qnet_target(next_obs)
            q_value_next = q_values_next.max(1)[0]

            td_target = reward + self.gamma * q_value_next * (1 - done)
        loss = self.loss(q_value, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size,device=torch.device('cpu')):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, next_obs, done):
        obs = obs if torch.is_tensor(obs) else torch.from_numpy(obs).float()
        action = action if torch.is_tensor(action) else torch.tensor(action).int()
        reward = reward if torch.is_tensor(reward) else torch.tensor(reward).int()
        next_obs = next_obs if torch.is_tensor(next_obs) else torch.from_numpy(next_obs).float()
        done = done if torch.is_tensor(done) else torch.tensor(done).int()
        self.buffer.append((obs, action, reward, next_obs, done))

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        obs = torch.stack([x[0] for x in data]).to(self.device)
        action = torch.stack([x[1] for x in data]).to(self.device)
        reward = torch.stack([x[2] for x in data]).to(self.device)
        next_obs = torch.stack([x[3] for x in data]).to(self.device)
        done = torch.stack([x[4] for x in data]).to(self.device)
        return obs, action, reward, next_obs, done

    def save(self, filepath):
        """保存经验缓冲区"""
        buffer_data = {
            'buffer': list(self.buffer),
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size
        }
        torch.save(buffer_data, filepath)
        print(f"Buffer saved to {filepath}, size: {len(self.buffer)}")

    def load(self, filepath):
        """加载经验缓冲区"""
        if os.path.exists(filepath):
            buffer_data = torch.load(filepath)
            self.buffer = deque(buffer_data['buffer'], maxlen=self.buffer_size)
            print(f"Buffer loaded from {filepath}, size: {len(self.buffer)}")
        else:
            print(f"No buffer file found at {filepath}, starting with empty buffer")

def train():
    #加载环境
    env = gym.make('Acrobot-v1')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = 3
    agent = Acrobot(obs_dim, act_dim,device=device,epsilon=0.5,epsilon_end=0.1,lr=0.0001)
    #agent.qnet.load_state_dict(torch.load("./models/acrobot_actor_V1.pth")) #接着训练（没有数据时注释掉）
    #agent.buffer.load("./models/acrobot_buffer_500.pth")
    episodes = 501
    aver_reward = 0
    print_num = 10
    sync_num = 20
    agent.qnet.train()

    for episode in range(episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            aver_reward += reward

        if episode % print_num == 0 and episode != 0:
            print("Episode:", episode, "Reward:", aver_reward/print_num)
            aver_reward = 0

        if episode % sync_num == 0:
            agent.sync_target()

        if episode % 500 == 0 and episode != 0:
            torch.save(agent.qnet.state_dict(), "./models/acrobot_qnet-{}.pth".format(episode))#保存模型参数
            agent.buffer.save('./models/acrobot_buffer_{}.pth'.format(episode))#保存经验缓冲区

def test():
    env = gym.make('Acrobot-v1',render_mode='human')
    obs_dim = env.observation_space.shape[0]
    act_dim = 3
    agent = Acrobot(obs_dim, act_dim,epsilon=0,epsilon_end=0)
    agent.qnet.load_state_dict(torch.load("../../saved_models/acrobot_actor_V1.pth"))  #加载模型参数
    agent.qnet.eval()
    total_reward = 0
    done = False
    obs, info = env.reset()

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        obs = next_obs
        total_reward += reward
    print("Total reward:", total_reward)


if __name__ == "__main__":
    #train()
    test()