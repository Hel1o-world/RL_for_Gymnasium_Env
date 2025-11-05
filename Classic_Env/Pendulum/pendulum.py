import torch
from PPO import PPO
from Data_Generate import DataGenerator
import gymnasium as gym

params = {
    'obs_dim': 3,#观测维度
    'act_dim': 1,#动作维度
    'actor_lr':3e-4,#actor网络学习率
    'critic_lr':3e-4,#critic网络学习率
    'epoch_per_update': 10,#采完一个批量的数据后更新多少次
    'gamma': 0.99,#折扣因子
    'sigma': 0.5,#固定的标准差
    'epsilon': 0.2,#截断系数
    'batch_size': 1024,#批量大小
    'weight_v': 0.5,#l_value的权重
    'max_episodes': 1000,#进行多少次外层迭代
    'step_per_batch': 4000,#一个批量数据采集多少步长
    'save_per_episode': 10,#多少回合保存一次
}

def train():
    env = gym.make('Pendulum-v1')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PPO(obs_dim=params['obs_dim'], act_dim=params['act_dim'], device=device, actor_lr=params['actor_lr'],
                critic_lr=params['critic_lr'], gamma=params['gamma'], sigma=params['sigma'], epochs=params['epoch_per_update'],
                batch_size=params['batch_size'], weight_v=params['weight_v'])
    generator = DataGenerator(env=env, agent=agent, steps_per_batch=params['step_per_batch'], gamma=params['gamma'],device=device)

    agent.actor.train()
    agent.critic.train()

    for episode in range(params['max_episodes']):
        train_dataset, total_reward = generator.generate_data()
        agent.update(train_dataset)
        print('Episode: {}, Total reward: {}'.format(episode, total_reward))

        if episode % params['epoch_per_update'] == 0:
            torch.save(agent.actor.state_dict(), './models/actor.pth')
            torch.save(agent.critic.state_dict(), './models/critic.pth')

def test():
    params['step_per_batch'] = 200
    params['sigma'] = 0.01
    env = gym.make('Pendulum-v1',render_mode='human')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = PPO(obs_dim=params['obs_dim'], act_dim=params['act_dim'], device=device, actor_lr=params['actor_lr'],
                critic_lr=params['critic_lr'], gamma=params['gamma'], sigma=params['sigma'],
                epochs=params['epoch_per_update'],
                batch_size=params['batch_size'], weight_v=params['weight_v'])
    generator = DataGenerator(env=env, agent=agent, steps_per_batch=params['step_per_batch'], gamma=params['gamma'],device=device)
    agent.actor.eval()
    agent.critic.eval()
    agent.actor.load_state_dict(torch.load('../../saved_models/pendulum_actor_V2.pth'))
    test_dataset, total_reward = generator.generate_data()
    print('Total reward: {}'.format(total_reward))


if __name__ == '__main__':
    #train()
    test()