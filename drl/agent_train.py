'''
环境中当前时刻的排水量等于上一个时刻的排水量

'''
import os
import warnings
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray

from drl.dqn_agent import DQNAgent
from drl.gas_extraction_env import GasExtractionEnv
from drl.qnetwork import QNetwork
from utils import date_util
from utils.my_constant import create_cnn_lstm_surrogate_model
from config import config

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))

def train():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 为了避免负号等特殊符号显示异常，添加下面这行
    plt.rcParams['axes.unicode_minus'] = False

    warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
    timestr = date_util.get_today_str()
    pth_path = os.path.join(project_root, "surrogate_model\weights", f"cnn_lstm_model_parameters_{timestr}.pth")
    model = create_cnn_lstm_surrogate_model(pth_path)

    # 初始化环境
    env = GasExtractionEnv(model)

    # 获取状态和动作空间的维度
    state_size = np.prod(env.observation_space.shape)  # 将状态展平为一维
    action_size = env.action_space.nvec[0]  # 每口井的动作空间大小（3）

    # 初始化DQN代理
    agent = DQNAgent(state_size, action_size, QNetwork(state_size, 128, action_size))

    # 用于存储每一轮训练的总奖励，方便后续可视化
    episode_rewards = []

    # 训练过程
    episodes = 1000
    for e in range(episodes):
        state = env.reset()
        state = state.flatten()  # 展平状态为一维向量
        total_reward = 0
        done = False
        while not done:
            # 选择动作
            action = agent.act(state)

            # 执行动作并得到反馈
            next_state, reward, done, info = env.step(action)
            next_state = next_state.flatten()  # 展平状态

            # 记住经验
            agent.remember(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            total_reward += reward

            # 训练
            agent.replay()

        # 每隔一定步数更新目标网络
        if e % 10 == 0:
            agent.update_target_model()

        episode_rewards.append(total_reward)
        print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    # 绘制训练奖励曲线
    plt.plot(range(1, episodes + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Episodes')
    plt.show()


    # 保存最终模型
    model_save_path = os.path.join(current_dir, "weights", "dqn_gas_extraction_model_final_r2.pth")
    torch.save(agent.model.state_dict(), model_save_path)
    print("模型保存成功了")

if __name__ == '__main__':
    train()
