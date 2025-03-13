import csv
import os
import random
import time
import warnings
from collections import deque
from datetime import datetime

import gym
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gym import spaces
from gym.spaces import Discrete, MultiDiscrete, Box
from numpy import ndarray
from torch import optim, nn, Tensor
from config import config
from utils import date_util
from utils.my_constant import create_cnn_lstm_surrogate_model
from mpl_toolkits.mplot3d import Axes3D  # 用于三维绘图

# 配置文件路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
# 加载数据标准化器
scaler_input = joblib.load(os.path.join(project_root, r"scaler\scaler_input.pkl"))
scaler_output = joblib.load(os.path.join(project_root, r"scaler\scaler_output.pkl"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei']
# 为了避免负号等特殊符号显示异常，添加下面这行
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# 定义状态空间的形状
state_shape: tuple = (
config.get("GasExtractionEnv.observation_space.shape_x"), config.get("GasExtractionEnv.observation_space.shape_y"))


class QNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state: Tensor) -> Tensor:
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


class DQNAgent:
    def __init__(self, state_size, action_size, model):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.model = model
        self.target_model = QNetwork(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)  # 调整学习率
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9999
        self.gamma = 0.88

        # 记录训练过程中的数据
        self.loss_records = []  # 记录每个 episode 的平均损失
        self.epsilon_records = []  # 记录探索率
        self.action_records = []  # 记录每个 episode 的动作分布

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state: ndarray) -> np.ndarray:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 3, size=17)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = state.view(state.size(0), -1)

        q_values = self.model(state)
        actions = []
        for i in range(17):
            action = torch.argmax(q_values[:, i * 3: (i + 1) * 3], dim=1).item()
            actions.append(action)

        return np.array(actions)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        losses = []
        for state, action, reward, next_state, done in batch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

            state = state.view(state.size(0), -1)
            next_state = next_state.view(next_state.size(0), -1)

            # 计算目标值
            target = reward + (1 - done) * self.gamma * torch.max(self.target_model(next_state), dim=1)[0].unsqueeze(1)
            target_f = self.model(state).clone()

            # 更新目标值
            for i in range(17):
                action_value = int(action[i])
                if action_value < 0 or action_value > 2:
                    action_value = 0
                target_f[0][i * 3 + action_value] = target

            # 计算损失
            loss = nn.MSELoss()(target_f, self.model(state))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            # 打印损失值和奖励值
            print(f"损失值: {loss.item()}, 奖励值: {reward.item()}")

        # 记录平均损失
        self.loss_records.append(np.mean(losses))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # 记录探索率
        self.epsilon_records.append(self.epsilon)


class GasExtractionEnv(gym.Env):
    def __init__(self, model):
        super(GasExtractionEnv, self).__init__()
        self.model = model
        self.time_steps: int = config.get("GasExtractionEnv.time_steps")
        self.current_time: int = 1
        self.action_space = MultiDiscrete([3] * 17)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32)
        self.state: ndarray = np.zeros(state_shape)
        self.max_drainage: int = config.get("GasExtractionEnv.max_drainage")
        self.min_drainage: int = config.get("GasExtractionEnv.min_drainage")
        self.drainage_records: list = []
        self.gas_production_records: list = []
        self.reward_records: list[float] = []
        self.reward_type: int = 1

        # 加载历史数据
        self.history_data = pd.read_csv(os.path.join(project_root, "data\\true\\history_data.csv"))
        self.history_drainage = self.history_data.iloc[:, 1:18].values
        self.history_time_steps = self.history_data.iloc[:, 0].values

        # 添加记录状态、动作、奖励的列表
        self.state_records = []
        self.action_records = []
        self.reward_records_detailed = []

    def reset(self) -> ndarray:
        self.current_time = 0
        initial_drainage = self.history_drainage[0] * (1 + np.random.uniform(-0.2, 0.2, size=17))
        initial_drainage = np.clip(initial_drainage, self.min_drainage, self.max_drainage)

        # 生成时间步，确保为正数
        self.state[:, 0] = np.array([self.current_time + i + 1 for i in range(4)])
        self.state[:, 1:] = np.vstack([initial_drainage] * 4)

        # 标准化处理
        self.state = scaler_input.transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0],
                                                                                            state_shape[1])

        # 记录初始状态
        self.state_records.append(self.state.copy())
        return self.state

    def step(self, action: np.ndarray):
        t: Tensor = torch.tensor(self.state, dtype=torch.float32)
        gas_before: Tensor = self.model(t.unsqueeze(0).to(device))
        gas_before: ndarray = gas_before.cpu().detach().numpy().flatten()

        state_original = scaler_input.inverse_transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0],
                                                                                                        state_shape[1])

        action_vector, action_explanation = self._decode_action(action)

        for i in range(17):
            new_drainage = state_original[-1, i + 1] * (1 + action_vector[i])
            new_drainage = np.clip(new_drainage, self.min_drainage, self.max_drainage)
            state_original[-1, i + 1] = new_drainage

        self.state = scaler_input.transform(state_original.reshape(-1, state_shape[1])).reshape(state_shape[0],
                                                                                                state_shape[1])

        gas_after = self.model(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device))
        gas_after = gas_after.cpu().detach().numpy().flatten()

        self.drainage_records.append(state_original[-1, 1:].tolist())
        self.gas_production_records.append(gas_after.tolist())

        reward = self.calculate_reward(gas_before, gas_after, action_vector)
        self.reward_records.append(reward)

        # 记录状态、动作、奖励
        self.state_records.append(self.state.copy())
        self.action_records.append(action.tolist())
        self.reward_records_detailed.append(reward)

        self.current_time += 1
        done: bool = self.current_time == self.time_steps

        if not done:
            state_original = self._update_state(state_original)
            self.state = scaler_input.transform(state_original.reshape(-1, state_shape[1])).reshape(state_shape[0],
                                                                                                    state_shape[1])

        return self.state, reward, done, {'action_explanation': action_explanation}

    def _update_state(self, state_original):
        state_original = np.roll(state_original, -1, axis=0)
        history_index = self.current_time % len(self.history_drainage)
        new_drainage = self.history_drainage[history_index] * (1 + np.random.uniform(-0.2, 0.2, size=17))
        new_drainage = np.clip(new_drainage, self.min_drainage, self.max_drainage)

        # 确保时间步为正数
        time_step_feature = np.array([[self.current_time + 4]])
        new_state_last_row = np.hstack((time_step_feature, new_drainage.reshape(1, -1)))
        new_state_last_row = scaler_input.transform(new_state_last_row.reshape(-1, state_shape[1])).reshape(1,
                                                                                                                state_shape[1])
        state_original[-1, :] = new_state_last_row[0, :]
        return state_original
    def _decode_action(self, action: np.ndarray) -> (list[float], list[str]):
        action_decoded = np.zeros(17)
        explanation = []
        for i in range(17):
            action_value = int(action[i])
            if action_value < 0 or action_value > 2:
                action_value = 0
            action_decoded[i] = [-0.2, 0, 0.2][action_value]
            explanation.append(
                f"井 {i + 1}: {'减少' if action_decoded[i] < 0 else '增加' if action_decoded[i] > 0 else '无变化'}")
        return action_decoded, explanation

    def calculate_reward(self, gas_before: ndarray[float], gas_after: ndarray[float],
                         action_vector: list[float]) -> float:
        gas_get: float = 1
        water_cost: float = 0.25
        gas_diff: float = sum(gas_after - gas_before)
        print(f"天然气产量变化: {gas_diff}")  # 打印 gas_diff
        reward_1: float = gas_diff * 1000  # 降低系数
        drainage_penalty = sum([action for action in action_vector]) * water_cost * 10  # 增加惩罚权重
        gas_reward = gas_diff * gas_get
        reward_2 = gas_reward - drainage_penalty
        penalty_for_drainage_change = sum([abs(action) for action in action_vector]) * 0.5  # 增加惩罚权重
        reward_3 = gas_diff * gas_get - penalty_for_drainage_change - drainage_penalty
        return reward_1

    def render(self):
        print(f"时间步 {self.current_time}")
        print(f"当前状态: {self.state}")
        print(f"执行动作: {self.action_records[-1]}")
        print(f"获得奖励: {self.reward_records_detailed[-1]}")
        print(f"天然气产量: {self.gas_production_records[-1]}")
        print(f"排水量: {self.drainage_records[-1]}")


def test_environment():
    timestr = date_util.get_today_str()
    pth_path = os.path.join(project_root, "surrogate_model\\weights", f"cnn_lstm_model_parameters_{timestr}.pth")
    model = create_cnn_lstm_surrogate_model(pth_path)

    env = GasExtractionEnv(model)
    state = env.reset()

    for step in range(env.time_steps):
        action = np.random.randint(0, 3, size=17)  # 随机动作
        next_state, reward, done, info = env.step(action)
        env.render()  # 输出当前状态、动作、奖励等信息

        if done:
            print("Episode 在 {} 步后结束".format(step + 1))
            break

    # 输出所有记录的状态、动作、奖励
    print("\n状态记录:")
    for i, state in enumerate(env.state_records):
        print(f"时间步 {i}: {state}")

    print("\n动作记录:")
    for i, action in enumerate(env.action_records):
        print(f"时间步 {i}: {action}")

    print("\n奖励记录:")
    for i, reward in enumerate(env.reward_records_detailed):
        print(f"时间步 {i}: {reward}")

    print("\n天然气产量记录:")
    for i, gas in enumerate(env.gas_production_records):
        print(f"时间步 {i}: {gas}")

    print("\n排水量记录:")
    for i, drainage in enumerate(env.drainage_records):
        print(f"时间步 {i}: {drainage}")


if __name__ == "__main__":
    test_environment()