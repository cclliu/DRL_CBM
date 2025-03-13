import os
import random
import pandas as pd
import numpy as np
import torch
import gym
from gym import spaces
from gym.spaces import Discrete, MultiDiscrete, Box
from numpy import ndarray
from torch import Tensor
import joblib
from config import config
from utils import date_util
from utils.my_constant import create_cnn_lstm_surrogate_model

# 加载配置和归一化器
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))

scaler_input = joblib.load(os.path.join(project_root, r"scaler\scaler_input.pkl"))
scaler_output = joblib.load(os.path.join(project_root, r"scaler\scaler_output.pkl"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_shape: tuple = (config.get("GasExtractionEnv.observation_space.shape_x"), config.get("GasExtractionEnv.observation_space.shape_y"))

class GasExtractionEnv(gym.Env):
    """
    煤层气排采强化学习环境
    """
    def __init__(self, model):
        super(GasExtractionEnv, self).__init__()
        self.model = model
        self.time_steps: int = config.get("GasExtractionEnv.time_steps")  # 总时间步数
        self.current_time: int = 0  # 当前时间步
        self.action_space = MultiDiscrete([3] * 17)  # 17 口井，每口井有 3 种操作
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32)  # 状态空间
        self.state: ndarray = np.zeros(state_shape)  # 初始化状态
        self.max_drainage: int = config.get("GasExtractionEnv.max_drainage")  # 定义排水量上限
        self.min_drainage: int = config.get("GasExtractionEnv.min_drainage")  # 定义排水量下限
        self.drainage_records: list = []  # 记录总排水量数据的列表
        self.gas_production_records: list = []  # 记录总产气量数据的列表
        self.reward_records: list[float] = []  # 记录每次获得的奖励数据的列表
        self.reward_type: int = 1

        # 加载历史数据
        self.history_data = pd.read_csv(os.path.join(project_root, "data\\true\\history_data.csv"))  # 假设历史数据文件为 history_data.csv
        self.history_drainage = self.history_data.iloc[:, 1:18].values  # 提取 2-18 列的排水量数据
        self.history_time_steps = self.history_data.iloc[:, 0].values  # 提取时间步数据

    def reset(self) -> ndarray:
        """
        重置环境状态，使用历史数据的扰动生成初始排水量数据
        """
        self.current_time = 0  # 重置时间步
        # 使用历史数据的第一个时间步的排水量数据，并加入随机扰动
        initial_drainage = self.history_drainage[0] * (1 + np.random.uniform(-0.1, 0.1, size=17))  # 扰动范围 ±10%
        initial_drainage = np.clip(initial_drainage, self.min_drainage, self.max_drainage)  # 确保数据在范围内

        # 初始化 state 数组
        self.state[:, 0] = np.array([self.current_time + i + 1 for i in range(4)])  # 时间序列
        self.state[:, 1:] = np.vstack([initial_drainage] * 4)  # 重复初始排水量数据 4 次

        # 归一化
        self.state = scaler_input.transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])
        return self.state

    def step(self, action: np.ndarray):
        """
        执行动作并返回新的状态、奖励、是否终止和其他信息
        """
        # 计算动作执行前的产气量
        t: Tensor = torch.tensor(self.state, dtype=torch.float32)
        gas_before: Tensor = self.model(t.unsqueeze(0).to(device))
        gas_before: ndarray = gas_before.cpu().detach().numpy().flatten()

        # 将标准化后的数据还原为原始数据
        self.state: ndarray = scaler_input.inverse_transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])

        # 解码动作，排水量调整比例
        action_vector, action_explanation = self._decode_action(action)

        # 更新排水量，按调整比例更新
        for i in range(17):
            self.state[-1, i + 1] *= (1 + action_vector[i])

        # 归一化
        self.state = scaler_input.transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])

        # 计算动作执行后的产气量
        gas_after = self.model(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device))
        gas_after = gas_after.cpu().detach().numpy().flatten()

        # 记录排水量和产气量
        self.drainage_records.append(self.state[-1, 1:].tolist())
        self.gas_production_records.append(gas_after.tolist())

        # 奖励计算
        reward: float = self.calculate_reward(gas_before, gas_after, action_vector)
        self.reward_records.append(reward)

        # 更新时间步
        self.current_time += 1
        done: bool = self.current_time == self.time_steps  # 达到最大时间步，结束

        # 移动时间步，并生成新的排水量
        if not done:
            # 将状态向前滚动
            self.state = np.roll(self.state, -1, axis=0)
            # 使用历史数据的对应时间步的排水量数据，并加入随机扰动
            history_index = self.current_time % len(self.history_drainage)  # 循环使用历史数据
            new_drainage = self.history_drainage[history_index] * (1 + np.random.uniform(-0.1, 0.1, size=17))  # 扰动范围 ±10%
            new_drainage = np.clip(new_drainage, self.min_drainage, self.max_drainage)  # 确保数据在范围内

            # 更新 state 的最后一行
            time_step_feature = np.array([[self.current_time + 4]])  # 当前时间步
            new_state_last_row = np.hstack((time_step_feature, new_drainage.reshape(1, -1)))
            new_state_last_row = scaler_input.transform(new_state_last_row.reshape(-1, state_shape[1])).reshape(1, state_shape[1])
            self.state[-1, :] = new_state_last_row[0, :]

        return self.state, reward, done, {'action_explanation': action_explanation}

    def _decode_action(self, action: np.ndarray) -> (list[float], list[str]):
        """
        将动作解码为每口井的操作调整比例

        参数:
        - action: 长度为17的动作向量，每个元素的值在 0、1、2 之间

        返回:
        - action_decoded: 每口井的调整比例（-0.1, 0, 0.1）
        - explanation: 每口井的操作解释
        """
        action_decoded = np.zeros(17)
        explanation = []
        for i in range(17):
            # 确保 action[i] 的值在 0、1、2 之间
            action_value = int(action[i])  # 转换为整数
            if action_value < 0 or action_value > 2:
                action_value = 0  # 如果超出范围，默认设置为 0
            action_decoded[i] = [-0.1, 0, 0.1][action_value]  # 动作解码为 -0.1, 0, 0.1
            explanation.append(
                f"Well {i + 1}: {'Decrease' if action_decoded[i] < 0 else 'Increase' if action_decoded[i] > 0 else 'No Change'}")
        return action_decoded, explanation

    def calculate_reward(self, gas_before: ndarray[float], gas_after: ndarray[float],action_vector: list[float]) -> float:
        """
        计算奖励。支持三种奖励方式：
        1. 以产气量变化作为奖励。
        2. 结合排水量变化和产气量变化进行奖励，排水量多为惩罚，产气量增加为奖励。
        3. 排水量增多为惩罚，产气量增加为奖励，排水量变化会给微小惩罚。
        """
        # 参数定义：生产气体的成本和生产水的成本
        gas_get: float = 1  # 生产 1 立方米气体的成本
        water_cost: float = 0.5  # 生产 1 立方米水的成本

        # 计算产气量的变化
        gas_diff: float = sum(gas_after - gas_before)

        # 奖励策略 1: 产气量变化作为奖励，奖励越多，产气量增加越多
        reward_1: float = gas_diff

        # 奖励策略 2: 排水量增加为惩罚，产气量增加为奖励
        drainage_penalty = sum([abs(action) for action in action_vector]) * water_cost  # 排水量的惩罚项
        gas_reward = gas_diff * gas_get  # 产气量的奖励项
        reward_2 = gas_reward - drainage_penalty

        # 奖励策略 3: 排水量增多为惩罚，产气量增加为奖励，改变了排水量也会给微小的惩罚
        penalty_for_drainage_change = sum([abs(action) for action in action_vector]) * 0.5  # 给排水量变化一个微小的惩罚
        reward_3 = gas_diff * gas_get - penalty_for_drainage_change - drainage_penalty

        # 返回选择的奖励策略
        return reward_1  # 可以选择 reward_1、reward_2 或 reward_3，根据需求

    def render(self):
        print(f"Time step {self.current_time}")
        print(f"Current state: {self.state}")

if __name__ == '__main__':

    # 初始化模型
    timestr = date_util.get_today_str()
    pth_path = os.path.join(project_root, "surrogate_model\weights", f"cnn_lstm_model_parameters_{timestr}.pth")
    model = create_cnn_lstm_surrogate_model(pth_path)
    # 初始化环境
    env = GasExtractionEnv(model)

    # 测试 reset 方法
    initial_state = env.reset()
    print("Initial State:")
    print(initial_state)

    # 测试 step 方法
    for _ in range(55):  # 测试 5 个时间步
        action = env.action_space.sample()  # 随机生成一个动作
        next_state, reward, done, info = env.step(action)
        print(f"Time Step {env.current_time}:")
        print("Action:", action)
        print("Next State:")
        print(next_state)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        env.render()
        if done:
            break