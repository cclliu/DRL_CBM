import os
import random
from collections import deque

import numpy as np
import torch
from numpy import ndarray
from torch import optim, nn

from drl.qnetwork import QNetwork

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
class DQNAgent:
    def __init__(self, state_size, action_size, model):
        """
        初始化DQN智能体。

        参数:
        - state_size: 状态空间的大小
        - action_size: 动作空间的大小
        - model: 智能体使用的Q网络模型实例
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.target_model = QNetwork(state_size, 128, action_size)  # 目标网络，隐藏层大小调整为128
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 使用Adam优化器，学习率调整为0.001
        self.memory = deque(maxlen=10000)  # 经验回放记忆，存储历史经验，最大长度为10000
        self.batch_size = 128  # 每次从记忆中采样进行训练的批量大小
        self.epsilon = 1.0  # 初始探索率，用于平衡探索新动作和利用已知最优动作
        self.epsilon_min = 0.01  # 探索率的最小值，避免探索率降为0导致完全停止探索
        self.epsilon_decay = 0.995  # 探索率的衰减率，控制探索率随训练轮数下降的速度
        self.gamma = 0.99  # 折扣因子，用于衡量未来奖励对当前价值的影响

        # 将目标网络初始化为与当前模型相同的参数
        self.update_target_model()

    def act(self, state):
        """
        根据当前状态选择一个动作，基于探索率决定是探索新动作还是利用已知最优动作。

        参数:
        - state: 当前环境状态

        返回:
        - 选择的动作，是一个长度为17的数组，每个元素表示对应井的操作（0, 1, 2）
        """
        if np.random.rand() <= self.epsilon:
            # 探索：随机生成一个动作向量，每个动作的值在 0、1、2 之间
            return np.random.randint(0, 3, size=17)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度
        q_values = self.model(state)  # 输出形状为 (1, action_size)
        return torch.argmax(q_values, dim=1).cpu().numpy()  # 利用：选择Q值最大的动作

    def remember(self, state, action, reward, next_state, done):
        """
        将当前的状态、动作、奖励、下一个状态以及是否结束的信息添加到经验回放记忆中。

        参数:
        - state: 当前状态
        - action: 执行的动作（长度为17的数组）
        - reward: 获得的奖励
        - next_state: 执行动作后的下一个状态
        - done: 是否结束的标志
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        从经验回放记忆中采样一批数据进行训练，更新Q网络的参数。
        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)  # actions 是 (batch_size, 17)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # 计算目标Q值
        with torch.no_grad():
            target_q_values = self.target_model(next_states).max(dim=1)[0]  # 取最大Q值
        targets = rewards + (1 - dones) * self.gamma * target_q_values

        # 计算当前Q值
        current_q_values = self.model(states)  # 输出形状为 (batch_size, action_size)

        # 由于 actions 是多维的，我们需要对每个动作计算 Q 值
        # 这里我们取每个动作对应的 Q 值的平均值
        action_q_values = torch.gather(current_q_values, 1, actions).mean(dim=1)

        # 计算损失并更新网络
        loss = nn.MSELoss()(action_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """
        将目标网络的参数更新为当前网络的参数，使目标网络跟上当前网络的变化，稳定训练过程。
        """
        self.target_model.load_state_dict(self.model.state_dict())