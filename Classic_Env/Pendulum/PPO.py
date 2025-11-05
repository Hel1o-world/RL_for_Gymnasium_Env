import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Actor(nn.Module): #策略网络
    def __init__(self,obs_dim, act_dim):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x #返回均值

class Critic(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x #返回状态值

class PPO:
    def __init__(self, obs_dim, act_dim,
                 actor_lr=1e-3,critic_lr=1e-3,
                 gamma=0.99,
                 epochs=10,
                 device='cpu',
                 sigma=0.1,
                 epsilon=0.2,
                 batch_size=64,
                 weight_v = 0.5
                 ):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.epochs = epochs
        self.device = device
        self.sigma = sigma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.weight_v = weight_v

    def get_action(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            mu = self.actor(obs)
            dist = torch.distributions.Normal(mu, self.sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob

    def update(self, train_dataset):
        for epoch in range(self.epochs):
            data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for _, (obs, action, old_log_prob, advantage, value_target) in enumerate(data_loader):
                mu = self.actor(obs)
                dist = torch.distributions.Normal(mu, self.sigma)
                new_log_prob = dist.log_prob(action) #计算新的动作概率
                ratio = torch.exp(new_log_prob.squeeze() - old_log_prob)
                l_clip = -torch.mean(torch.min(ratio * advantage,torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage))
                l_value = self.weight_v * F.mse_loss(self.critic(obs).squeeze(), value_target)

                self.actor_optimizer.zero_grad()
                l_clip.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                l_value.backward()
                self.critic_optimizer.step()


