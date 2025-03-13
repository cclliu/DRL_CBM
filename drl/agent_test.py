import numpy as np
import torch
import gym
from gym import spaces
from models.cnn_lstm import CNNLSTM
import joblib
import warnings
import torch.nn as nn
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
# 为了避免负号等特殊符号显示异常，添加下面这行
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

scaler_input = joblib.load(r'.\weight\scaler_input.pkl')
scaler_output = joblib.load(r'.\weight\scaler_output.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义煤层气排采强化学习环境类，与训练代码中的一致
class GasExtractionEnv(gym.Env):
    """
    煤层气排采强化学习环境
    """

    def __init__(self, model):
        super(GasExtractionEnv, self).__init__()
        self.model = model
        # self.time_steps = 33  # 总时间步数
        self.time_steps = 5  # 总时间步数
        # self.time_steps = 20  # 总时间步数
        self.current_time = 0  # 当前时间步
        self.action_space = spaces.Discrete(3 ** 5)  # 动作空间，3的5次方表示每个井口有3种操作
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 6), dtype=np.float32)  # 状态空间
        self.state = np.zeros((4, 6))  # 初始化状态
        self.max_drainage = 50  # 定义排水量上限属性，可根据实际调整
        self.min_drainage = 10  # 定义排水量下限属性，可根据实际调整
        self.drainage_records = []  # 记录总排水量数据的列表
        self.gas_production_records = []  # 记录总产气量数据的列表
        self.reward_records = []  # 记录每次获得的奖励数据的列表
        self.action_records = []  # 记录每步采取的动作
        self.action_explanation_records = []  # 记录每步动作解释
        self.step_info_records = []  # 新增，用于记录每个时间步详细信息

    def reset(self):
        """
        重置环境状态
        """
        self.current_time = 0  # 重置时间步
        # 初始化前四个时间步的排水量数据和时间步
        time_step = np.array([self.current_time + i + 1 for i in range(4)])  # 时间序列
        drainage = np.random.uniform(10, 50, size=(4, 5))  # 随机生成排采量
        self.state[:, 0] = time_step
        self.state[:, 1:] = drainage
        self.state = scaler_input.transform(self.state.reshape(-1, 6)).reshape(4, 6)  # 对状态进行归一化
        return self.state

    def step(self, action):
        """
        执行动作并返回新的状态、奖励、是否终止和其他信息
        """
        # 记录采取的动作
        self.action_records.append(action)

        # 解码动作并记录动作解释
        action_vector, action_explanation = self._decode_action(action)
        self.action_explanation_records.append(action_explanation)

        # 计算动作执行前的产气量
        gas_before = self.model(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device))
        gas_before = gas_before.cpu().detach().numpy().flatten()

        self.state = scaler_input.inverse_transform(self.state.reshape(-1, 6)).reshape(4, 6)

        # 检查并调整排水量，确保不超过最大值和最小值
        for i in range(5):
            current_drainage = self.state[-1, i + 1]
            new_drainage = current_drainage * (1 + action_vector[i])
            if new_drainage > self.max_drainage:
                if random.random() < 0.5:
                    action_vector[i] = 0
                else:
                    action_vector[i] = -0.1
            if new_drainage < self.min_drainage:
                if random.random() < 0.5:
                    action_vector[i] = 0
                else:
                    action_vector[i] = 0.1

        # 更新排水量，按调整比例更新
        for i in range(5):
            self.state[-1, i + 1] *= (1 + action_vector[i])

        # 归一化
        self.state = scaler_input.transform(self.state.reshape(-1, 6)).reshape(4, 6)

        # 计算动作执行后的产气量
        gas_after = self.model(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device))
        gas_after = gas_after.cpu().detach().numpy().flatten()

        # 记录排水量和产气量
        self.drainage_records.append(self.state[-1, 1:].tolist())
        self.gas_production_records.append(gas_after.tolist())

        # 奖励计算
        reward = self.calculate_reward(gas_before, gas_after, action_vector)

        self.reward_records.append(reward)

        # 更新时间步
        self.current_time += 1
        done = self.current_time == self.time_steps  # 达到最大时间步，结束

        # 记录当前时间步详细信息
        step_info = {
            "时间步": self.current_time,
            # "action前的当前状态": self.state.copy(),
            # "action前的产气量": gas_before.tolist(),
            "action前的当前状态": scaler_input.inverse_transform(self.state.reshape(-1, 6)).reshape(4, 6),
            "action前的产气量": scaler_output.inverse_transform(gas_before.reshape(-1, 5)).reshape(1, 5),
            "采取动作": action,
            "动作解释": action_explanation,
            # "action后的当前状态": self.state.copy(),
            # "action后的产气量": gas_after.tolist(),
            "action后的当前状态": scaler_input.inverse_transform(self.state.reshape(-1, 6)).reshape(4, 6),
            "action后的产气量": scaler_output.inverse_transform(gas_after.reshape(-1, 5)).reshape(1, 5),

            "获得奖励": reward

        }
        self.step_info_records.append(step_info)

        # 移动时间步，并生成新的排水量
        if not done:
            # 将状态向前滚动
            self.state = np.roll(self.state, -1, axis=0)
            # 生成新的排水量，并归一化
            new_drainage = np.random.uniform(10, 50, size=(1, 5))
            time_step_feature = np.array([[self.current_time + 4]])  # 当前时间步
            new_state_last_row = np.hstack((time_step_feature, new_drainage))
            new_state_last_row = scaler_input.transform(new_state_last_row.reshape(-1, 6)).reshape(1, 6)
            self.state[-1, :] = new_state_last_row[0, :]

            # 保证当前排水量与上一个时刻的排水量相同
            self.state[-1, 1:] = self.state[-2, 1:].copy()

        return self.state, reward, done, {'action_explanation': action_explanation}

    def _decode_action(self, action):
        """
        将动作解码为每口井的操作调整比例
        """
        action_decoded = np.zeros(5)
        explanation = []
        for i in range(5):
            action_decoded[i] = [-0.1, 0, 0.1][action % 3]  # 动作解码为-0.1, 0, 0.1
            explanation.append(
                f"Well {i + 1}: {'Decrease' if action_decoded[i] < 0 else 'Increase' if action_decoded[i] > 0 else 'No Change'}")
            action //= 3
        return action_decoded, explanation

    def calculate_reward(self, gas_before, gas_after, action_vector):
        """
        计算奖励。支持三种奖励方式：
        1. 以产气量变化作为奖励。
        2. 结合排水量变化和产气量变化进行奖励，排水量多为惩罚，产气量增加为奖励。
        3. 排水量增多为惩罚，产气量增加为奖励，排水量变化会给微小惩罚。
        """
        # 参数定义：生产气体的成本和生产水的成本
        gas_get = 1  # 生产1立方米气体的成本
        water_cost = 0.5  # 生产1立方米水的成本

        # 计算产气量的变化
        gas_diff = sum(gas_after - gas_before)

        # 奖励策略 1: 产气量变化作为奖励，奖励越多，产气量增加越多
        reward_1 = gas_diff

        # 奖励策略 2: 排水量增加为惩罚，产气量增加为奖励
        # drainage_penalty = sum([abs(action) for action in action_vector]) * water_cost  # 排水量的惩罚项，排水量增加为惩罚
        drainage_penalty = sum([action for action in action_vector]) * water_cost
        gas_reward = gas_diff * gas_get  # 产气量的奖励项，产气量增加为奖励

        reward_2 = gas_reward - drainage_penalty

        # 奖励策略 3: 排水量增多为惩罚，产气量增加为奖励，改变了排水量也会给微小的惩罚
        # 排水量增多的惩罚与产气量增加的奖励，排水量增加给微小惩罚
        penalty_for_drainage_change = sum([abs(action) for action in action_vector]) * 0.5  # 给排水量变化一个微小的惩罚
        reward_3 = gas_diff * gas_get - penalty_for_drainage_change - drainage_penalty

        # 返回选择的奖励策略，这里以reward_2为默认策略
        # 你可以通过环境中的某种机制来选择不同的奖励策略
        return reward_2  # 可以选择reward_1、reward_2 或 reward_3，根据需求

    def render(self):
        print(f"Time step {self.current_time}")
        print(f"Current state: {self.state}")


# 定义Q网络类，与训练代码中的一致
class QNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        """
        初始化Q网络，用于估计状态动作价值（Q值）。
        参数:
        - state_size: 状态空间的大小（输入层节点数）
        - hidden_size: 隐藏层的节点数量
        - action_size: 动作空间的大小（输出层节点数）
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        前向传播函数，定义数据在网络中的流动过程。
        参数:
        - state: 输入的状态张量
        返回:
        - 经过网络计算后的Q值张量，形状为(1, action_size)，表示对应各个动作的Q值估计
        """
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


# 创建环境
model = CNNLSTM(input_size=6, hidden_size=30, num_layers=3, output_size=5)
model.load_state_dict(torch.load(r'.\weight\cnn_lstm_model_parameters_2024.12.6.pth'))
model.to(device)
model.eval()
env = GasExtractionEnv(model)

# 获取状态和动作空间的维度
state_size = np.prod(env.observation_space.shape)  # 将4x6状态展平为一维
action_size = env.action_space.n  # 动作空间大小（243）

# 初始化Q网络并加载训练好的模型参数
q_network = QNetwork(state_size, 64, action_size)
q_network.load_state_dict(torch.load(r".\weight\r2_lesstep\dqn_gas_extraction_model1759.pth"))

q_network.to(device)
q_network.eval()

# 用于存储测试过程中的总奖励、每轮详细奖励、每轮动作记录等
test_episode_rewards = []
test_episode_reward_details = []
test_episode_action_records = []
test_episode_action_explanation_records = []

# 进行多次测试（这里以10次为例，可根据实际需求调整次数）
test_episodes = 1
for _ in range(test_episodes):
    state = env.reset()
    state = state.flatten()
    total_reward = 0
    rewards_per_episode = []
    actions_per_episode = []
    action_explanations_per_episode = []
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done, info = env.step(action)
        next_state = next_state.flatten()

        state = next_state
        total_reward += reward
        rewards_per_episode.append(reward)
        actions_per_episode.append(action)
        action_explanations_per_episode.append(info['action_explanation'])

    test_episode_rewards.append(total_reward)
    test_episode_reward_details.append(rewards_per_episode)
    test_episode_action_records.append(actions_per_episode)
    test_episode_action_explanation_records.append(action_explanations_per_episode)

    # 输出当前轮次每个时间步详细信息
    print(f"第 {_ + 1} 轮测试详细信息：")
    for step_info in env.step_info_records:
        for key, value in step_info.items():
            print(f"{key}: {value}")
        print("-" * 50)

# 输出测试结果相关统计信息
print("测试结果统计：")
print(f"平均总奖励：{np.mean(test_episode_rewards)}")
print(f"总奖励最小值：{np.min(test_episode_rewards)}")
print(f"总奖励最大值：{np.max(test_episode_rewards)}")

# 找到最高奖励的轮次索引
max_reward_index = np.argmax(test_episode_rewards)
print(f"最高奖励的轮次为：第 {max_reward_index + 1} 轮")
print(f"该轮采取的动作编码记录：{test_episode_action_records[max_reward_index]}")
print(f"该轮采取的动作解释记录：{test_episode_action_explanation_records[max_reward_index]}")

# 输出每一轮每一步的排水量和产气量具体值（按照完整时间步顺序展示，以最高奖励轮次为例，可按需扩展到其他轮次）
print(f"最高奖励轮次（第 {max_reward_index + 1} 轮）完整时间步的排水量具体值：")
# 调整展示方式，按照期望的完整时间步顺序输出排水量
for i in range(len(env.drainage_records)):
    print(f"时间步 {i + 1}: {env.drainage_records[i]}")
print(f"最高奖励轮次（第 {max_reward_index + 1} 轮）完整时间步的产气量具体值：")
# 调整展示方式，按照期望的完整时间步顺序输出产气量
for i in range(len(env.gas_production_records)):
    print(f"时间步 {i + 1}: {env.gas_production_records[i]}")

# 绘制每一轮的总奖励变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, test_episodes + 1), test_episode_rewards)
plt.xlabel('测试轮次')
plt.ylabel('总奖励')
plt.title('测试过程总奖励变化情况')
plt.legend(['总奖励'])
plt.show()

# 绘制最高奖励轮次的每步奖励变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(test_episode_reward_details[max_reward_index])), test_episode_reward_details[max_reward_index])
plt.xlabel('时间步')
plt.ylabel('每步奖励')
plt.title(f'最高奖励轮次（第 {max_reward_index + 1} 轮）每步奖励变化情况')
plt.legend(['每步奖励'])
plt.show()

# 绘制最高奖励轮次的排水量变化曲线（以第一口井为例，可按需修改查看其他井）
plt.figure(figsize=(10, 6))
first_well_drainage = [step[0] for step in env.drainage_records if step]
plt.plot(range(len(first_well_drainage)), first_well_drainage)
plt.xlabel('时间步')
plt.ylabel('第一口井排水量')
plt.title(f'最高奖励轮次（第 {max_reward_index + 1} 轮）第一口井排水量变化情况')
plt.legend(['第一口井排水量'])
plt.show()

# 绘制最高奖励轮次的所有井排水量变化曲线
plt.figure(figsize=(10, 6))
num_wells = len(env.drainage_records[0])  # 假设每个时间步都有相同数量的井数据
for i in range(num_wells):
    well_drainage = [step[i] for step in env.drainage_records if step]
    plt.plot(range(len(well_drainage)), well_drainage, label=f'第 {i + 1} 口井排水量')

plt.xlabel('时间步')
plt.ylabel('排水量')
plt.title(f'最高奖励轮次（第 {max_reward_index + 1} 轮）所有井排水量变化情况')
plt.legend()
plt.show()

# 绘制最高奖励轮次的产气量变化曲线
plt.figure(figsize=(10, 6))
gas_production = env.gas_production_records if env.gas_production_records else [0] * len(env.drainage_records)
plt.plot(range(len(gas_production)), gas_production)
plt.xlabel('时间步')
plt.ylabel('产气量')
plt.title(f'最高奖励轮次（第 {max_reward_index + 1} 轮）产气量变化情况')
plt.legend(['产气量'])
plt.show()