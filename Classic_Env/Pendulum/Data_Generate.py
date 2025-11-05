import torch
import numpy as np
'使用当前actor网络、critic网络生成数据集（obs, action, old_log_prob, advatage, v_target）,供PPO进行训练'

class DataGenerator:
    def __init__(self, env, agent,
                 steps_per_batch=2000,
                 gamma=0.99,
                 device='cpu',):
        self.env = env
        self.agent = agent
        self.step_per_batch = steps_per_batch
        self.gamma = gamma
        self.device = device
        self.train_data = []

    def generate_data(self):
        batch_obs = []
        batch_acts = []
        batch_rews = []
        batch_old_log_probs = []

        t = 0

        while t < self.step_per_batch:
            episode_rews = []
            obs, _ = self.env.reset()
            done = False
            while not done:
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.agent.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated | truncated
                episode_rews.append(rew)
                batch_acts.append(action)
                batch_old_log_probs.append(log_prob)
            batch_rews.append(episode_rews)
        total_rew = sum(sum(episode_rews) for episode_rews in batch_rews)/len(batch_rews)
        batch_value_targets = self.compute_value_target(batch_rews).to(self.device)
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).to(self.device)
        batch_old_log_probs = torch.tensor(batch_old_log_probs, dtype=torch.float).to(self.device)
        batch_advantages = self.compute_advantage(batch_obs, batch_value_targets).to(self.device)
        train_dataset = torch.utils.data.TensorDataset(batch_obs, batch_acts, batch_old_log_probs, batch_advantages, batch_value_targets)
        return train_dataset, total_rew

    def compute_value_target(self, batch_rews):
        """逆向计算当前状态的状态值，作为critic网络训练的目标"""
        batch_value_targets = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for reward in reversed(ep_rews):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_value_targets.insert(0, discounted_reward)
        batch_value_targets = torch.tensor(batch_value_targets, dtype=torch.float)
        return batch_value_targets

    def compute_advantage(self, batch_obs, batch_value_targets):
        """计算优势函数"""
        batch_values = self.agent.critic(batch_obs).squeeze()
        batch_advantages = batch_value_targets - batch_values
        batch_advantages = batch_advantages.detach()
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-10) #标准化，稳定训练
        return batch_advantages
