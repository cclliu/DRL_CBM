'''目前最终版本2025.1.131111111111'''

import os
import random
import warnings
from collections import deque
import gym
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gym.spaces import MultiDiscrete, Box
from numpy import ndarray
from torch import optim, nn, Tensor
from config import config
from utils import date_util
from utils.my_constant import create_cnn_lstm_surrogate_model

# 配置文件路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载数据标准化器
scaler_input = joblib.load(os.path.join(project_root, r"scaler\scaler_input.pkl"))
scaler_output = joblib.load(os.path.join(project_root, r"scaler\scaler_output.pkl"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置全局字体为新罗马
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示异常
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# 定义状态空间的形状
state_shape: tuple = (
    config.get("GasExtractionEnv.observation_space.shape_x"), config.get("GasExtractionEnv.observation_space.shape_y")
)

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 调整学习率
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
        actions = [torch.argmax(q_values[:, i * 3: (i + 1) * 3], dim=1).item() for i in range(17)]
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
        self.current_time: int = 0  # 当前时间步
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
        self.history_drainage = self.history_data.iloc[:, 1:18].values  # 第1-17列是排水量
        self.history_time_steps = self.history_data.iloc[:, 0].values  # 第0列是时间步

        # 添加记录状态、动作、奖励的列表
        self.state_records = []
        self.action_records = []
        self.reward_records_detailed = []

    def reset(self) -> ndarray:
        self.current_time = 0  # 重置当前时间步
        initial_drainage = self.history_drainage[0] * (1 + np.random.uniform(-0.2, 0.2, size=17))
        initial_drainage = np.clip(initial_drainage, self.min_drainage, self.max_drainage)

        # 初始化状态，时间列从当前时间步开始递增
        self.state[:, 0] = np.array([self.current_time + i + 1 for i in range(state_shape[0])])
        self.state[:, 1:] = np.vstack([initial_drainage] * state_shape[0])
        # 标准化状态
        self.state = scaler_input.transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])
        # 记录初始状态
        self.state_records.append(self.state.copy())
        return self.state

    def step(self, action: np.ndarray):
        t: Tensor = torch.tensor(self.state, dtype=torch.float32)
        gas_before: Tensor = self.model(t.unsqueeze(0).to(device))
        gas_before: ndarray = gas_before.cpu().detach().numpy().flatten()

        # 反标准化状态以更新排水量
        state_original = scaler_input.inverse_transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])

        # 解码动作
        action_vector, action_explanation = self._decode_action(action)

        # 更新排水量
        for i in range(17):
            new_drainage = state_original[-1, i + 1] * (1 + action_vector[i])
            new_drainage = np.clip(new_drainage, self.min_drainage, self.max_drainage)
            state_original[-1, i + 1] = new_drainage

        # 标准化更新后的状态
        self.state = scaler_input.transform(state_original.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])

        # 计算更新后的天然气产量
        gas_after = self.model(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device))
        gas_after = gas_after.cpu().detach().numpy().flatten()

        # 记录排水量和天然气产量
        self.drainage_records.append(state_original[-1, 1:].tolist())
        self.gas_production_records.append(gas_after.tolist())

        # 计算奖励
        reward: float = self.calculate_reward(gas_before, gas_after, action_vector)
        self.reward_records.append(reward)

        # 记录状态、动作、奖励
        self.state_records.append(self.state.copy())
        self.action_records.append(action.tolist())
        self.reward_records_detailed.append(reward)

        # 更新时间步
        self.current_time += 1
        done: bool = self.current_time == self.time_steps

        if not done:
            # 移除最上面的一行
            state_original = np.delete(state_original, 0, axis=0)

            # 从历史数据中获取新的排水量
            history_index = self.current_time % len(self.history_drainage)
            new_drainage = self.history_drainage[history_index] * (1 + np.random.uniform(-0.2, 0.2, size=17))
            new_drainage = np.clip(new_drainage, self.min_drainage, self.max_drainage)

            # 更新时间列
            time_step_feature = np.array([[self.current_time + state_shape[0] - 1]])  # 时间列递增
            new_state_last_row = np.hstack((time_step_feature, new_drainage.reshape(1, -1)))

            # 添加新的一行
            state_original = np.vstack((state_original, new_state_last_row))

            # 标准化新的一行并更新状态
            self.state = scaler_input.transform(state_original.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])

        return self.state, reward, done, {'action_explanation': action_explanation}

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

    def calculate_reward(self, gas_before: ndarray[float], gas_after: ndarray[float], action_vector: list[float]) -> float:
        gas_get: float = 1
        water_cost: float = 0.1
        gas_diff: float = sum(gas_after - gas_before)
        print(f"天然气产量变化: {gas_diff}")  # 打印 gas_diff
        reward_1: float = gas_diff * 1  # 降低系数
        drainage_penalty = sum([action for action in action_vector]) * water_cost  # 增加惩罚权重
        gas_reward = gas_diff * gas_get
        reward_2 = gas_reward - drainage_penalty
        penalty_for_drainage_change = sum([abs(action) for action in action_vector])   # 增加惩罚权重
        reward_3 = gas_diff * gas_get - penalty_for_drainage_change - drainage_penalty
        return reward_2

    def render(self):
        # 反标准化当前状态以获取原始的排水量
        state_original = scaler_input.inverse_transform(self.state.reshape(-1, state_shape[1])).reshape(state_shape[0], state_shape[1])
        original_drainage = state_original[-1, 1:]  # 获取最新的排水量（原始值）
        original_time = state_original[-1, 0]  # 获取最新的时间列（原始值）

        # 计算原始的产气量
        t = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(device)
        gas_production_normalized = self.model(t).cpu().detach().numpy().flatten()  # 归一化后的产气量
        gas_production_original = scaler_output.inverse_transform(gas_production_normalized.reshape(1, -1)).flatten()  # 反标准化后的产气量

        # 输出当前时间步
        print(f"时间步 {self.current_time}")

        # 输出归一化前后的排水量（带上时间列）
        print("\n归一化后的状态（时间列 + 排水量）:")
        print(self.state[-1, :])  # 归一化后的状态（时间列 + 排水量）

        print("\n原始状态（时间列 + 排水量）:")
        print(state_original[-1, :])  # 原始状态（时间列 + 排水量）

        # 输出归一化前后的产气量
        print("\n归一化后的产气量:")
        print(gas_production_normalized)  # 归一化后的产气量
        print("\n原始产气量:")
        print(gas_production_original)  # 原始产气量

        # 输出执行的动作
        print(f"\n执行动作: {self.action_records[-1]}")

        # 输出获得的奖励
        print(f"获得奖励: {self.reward_records_detailed[-1]}")

def save_records(env, episode, save_dir, episode_losses, episode_rewards):
    """
    保存状态、动作、奖励、损失函数和奖励函数的值到 txt 和 excel 文件
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存状态记录
    state_records_path = os.path.join(save_dir, f"state_records_episode_{episode}.txt")
    with open(state_records_path, "w") as f:
        for i, state in enumerate(env.state_records):
            f.write(f"时间步 {i}:\n{state}\n\n")

    # 保存动作记录
    action_records_path = os.path.join(save_dir, f"action_records_episode_{episode}.txt")
    with open(action_records_path, "w") as f:
        for i, action in enumerate(env.action_records):
            f.write(f"时间步 {i}: {action}\n")

    # 保存奖励记录
    reward_records_path = os.path.join(save_dir, f"reward_records_episode_{episode}.txt")
    with open(reward_records_path, "w") as f:
        for i, reward in enumerate(env.reward_records_detailed):
            f.write(f"时间步 {i}: {reward}\n")

    # 保存天然气产量记录
    gas_production_records_path = os.path.join(save_dir, f"gas_production_records_episode_{episode}.txt")
    with open(gas_production_records_path, "w") as f:
        for i, gas in enumerate(env.gas_production_records):
            f.write(f"时间步 {i}: {gas}\n")

    # 保存排水量记录
    drainage_records_path = os.path.join(save_dir, f"drainage_records_episode_{episode}.txt")
    with open(drainage_records_path, "w") as f:
        for i, drainage in enumerate(env.drainage_records):
            f.write(f"时间步 {i}: {drainage}\n")

    # 保存损失函数记录
    loss_records_path = os.path.join(save_dir, f"loss_records_episode_{episode}.txt")
    with open(loss_records_path, "w") as f:
        for i, loss in enumerate(episode_losses):
            f.write(f"Episode {i + 1}: {loss}\n")

    # 保存奖励函数记录
    reward_function_records_path = os.path.join(save_dir, f"reward_function_records_episode_{episode}.txt")
    with open(reward_function_records_path, "w") as f:
        for i, reward in enumerate(episode_rewards):
            f.write(f"Episode {i + 1}: {reward}\n")

    # 保存到 Excel 文件
    excel_path = os.path.join(save_dir, f"records_episode_{episode}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        # 保存状态记录（展平为二维数组）
        if len(env.state_records) > 0:
            state_records_flat = np.array(env.state_records).reshape(len(env.state_records), -1)
            pd.DataFrame(state_records_flat).to_excel(writer, sheet_name="状态记录", index=False)

        # 保存动作记录
        if len(env.action_records) > 0:
            pd.DataFrame(env.action_records).to_excel(writer, sheet_name="动作记录", index=False)

        # 保存奖励记录
        if len(env.reward_records_detailed) > 0:
            pd.DataFrame(env.reward_records_detailed, columns=["奖励"]).to_excel(writer, sheet_name="奖励记录", index=False)

        # 保存天然气产量记录
        if len(env.gas_production_records) > 0:
            pd.DataFrame(env.gas_production_records).to_excel(writer, sheet_name="天然气产量记录", index=False)

        # 保存排水量记录
        if len(env.drainage_records) > 0:
            pd.DataFrame(env.drainage_records).to_excel(writer, sheet_name="排水量记录", index=False)

        # 保存损失函数记录
        if len(episode_losses) > 0:
            pd.DataFrame(episode_losses, columns=["损失"]).to_excel(writer, sheet_name="损失函数记录", index=False)

        # 保存奖励函数记录
        if len(episode_rewards) > 0:
            pd.DataFrame(episode_rewards, columns=["奖励"]).to_excel(writer, sheet_name="奖励函数记录", index=False)

def train():
    timestr = date_util.get_today_str()
    pth_path = os.path.join(project_root, "surrogate_model\\weights", f"cnn_lstm_model_parameters_{timestr}.pth")
    model = create_cnn_lstm_surrogate_model(pth_path)

    env = GasExtractionEnv(model)
    state_size = np.prod(env.observation_space.shape)
    action_size = 17 * 3  # 每个井有3种动作（减少、不变、增加），共17个井
    agent = DQNAgent(state_size, action_size, QNetwork(state_size, 128, action_size))

    episode_rewards = []
    episodes =2001

    # 创建 plots 文件夹（如果不存在）
    plots_dir = os.path.join(project_root, "plots_figures")
    os.makedirs(plots_dir, exist_ok=True)

    # 创建 records 文件夹（如果不存在）
    records_dir = os.path.join(project_root, "records")
    os.makedirs(records_dir, exist_ok=True)

    # 用于记录每个回合的平均损失
    episode_losses = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []  # 记录当前 episode 的动作分布
        episode_loss = []  # 记录当前 episode 的损失值

        while not done:
            action = agent.act(state)  # 获取动作
            next_state, reward, done, info = env.step(action)  # 执行动作
            agent.remember(state, action, reward, next_state, done)  # 存储经验
            state = next_state
            total_reward += reward
            agent.replay()  # 训练模型

            # 记录当前 episode 的损失值
            if len(agent.loss_records) > 0:
                episode_loss.append(agent.loss_records[-1])

            # 记录当前 episode 的动作
            episode_actions.append(action)

        # 记录当前 episode 的动作分布
        agent.action_records.append(episode_actions)

        # 计算当前 episode 的平均损失
        if len(episode_loss) > 0:
            avg_loss = np.mean(episode_loss)
            episode_losses.append(avg_loss)
        else:
            episode_losses.append(0.0)  # 如果没有损失值，记录为 0

        # 每 100 个 episode 更新目标网络并保存记录
        if e % 100 == 0:
            agent.update_target_model()
            # 保存状态、动作、奖励等记录
            save_records(env, e + 1, records_dir, episode_losses, episode_rewards)

            # 保存状态、动作、奖励等记录
            # save_records(env, e + 1, records_dir)

            # 每 100 个 episode 输出一次动作分布三维图
            # 定义井名列表
            well_names = [
                "TS-010", "TS46-02", "TS46-03", "TS47-04", "TS47-05", "TS56-01", "TS56-02",
                "TS56-03", "TS56-04", "TS56-08", "TS5603D1", "TS57-01", "TS57-02", "TS57-03",
                "TS57-07", "TS57-09", "TS58-01"
            ]

            # 每 100 个 episode 输出一次动作分布三维图
            if e % 200 == 0:
                fig = plt.figure(figsize=(14, 8))  # 增加画布宽度
                ax = fig.add_subplot(111, projection='3d')
                episode_actions = np.array(agent.action_records[-1])  # 获取当前 episode 的动作

                # 确保时间步为正数
                time_steps = np.arange(len(episode_actions))  # 时间步从 0 开始
                if np.any(time_steps < 0):
                    raise ValueError("Time steps contain negative values!")

                # 将动作值映射为数值
                action_mapping = {'Decrease': -1, 'No Change': 0, 'Increase': 1}
                for i in range(17):
                    # 将动作值映射为具体的动作
                    action_labels = ['Decrease', 'No Change', 'Increase']
                    actions_mapped = [action_mapping[action_labels[action]] for action in episode_actions[:, i]]
                    ax.scatter(time_steps, [i] * len(episode_actions), actions_mapped, label=well_names[i])  # 使用井名作为标签

                # 设置 z 轴的刻度和标签
                ax.set_zticks([-1, 0, 1])
                ax.set_zticklabels(['Decrease', '   No Change', 'Increase'])

                # 设置坐标轴标签
                ax.set_xlabel('Time Steps', fontsize=12, labelpad=10, fontname='Times New Roman')
                ax.set_ylabel('Well Index', fontsize=12, labelpad=10, fontname='Times New Roman')
                ax.set_zlabel('Action', fontsize=12, labelpad=10, fontname='Times New Roman')
                ax.set_title(f'Action Distribution over Time Steps (Episode {e + 1})', fontsize=14, pad=20,
                             fontname='Times New Roman')

                # 调整图例位置，进一步向右移动
                legend = ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10, title="Wells",
                                   title_fontsize=12, prop={'family': 'Times New Roman'})
                legend.get_frame().set_edgecolor('black')  # 设置图例边框颜色
                legend.get_frame().set_facecolor('white')  # 设置图例背景颜色

                # 调整视角
                ax.view_init(elev=15, azim=160)  # 设置仰角和方位角

                # 调整布局，确保图例不覆盖图形
                plt.tight_layout()
                plt.grid(True)
                plt.tick_params(axis='both', which='major', labelsize=10, width=2)  # 设置刻度值字体大小
                for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                    tick.set_fontname('Times New Roman')  # 设置刻度值字体
                    tick.set_fontweight('bold')  # 加粗刻度值字体

                # 保存图片
                plot_path = os.path.join(plots_dir, f"action_distribution_episode_{e + 1}.png")
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)  # 保存时调整边界，提高分辨率
                plt.close(fig)  # 关闭图像，避免内存泄漏

        # 每 100 个 episode 保存一次模型
        if e % 200 == 0:
            model_save_path = os.path.join(current_dir, "weights", f"dqn_gas_extraction_model_episode_r2_{e}.pth")
            torch.save(agent.model.state_dict(), model_save_path)

        # 记录当前 episode 的平均奖励
        episode_rewards.append(total_reward / env.current_time)
        print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    #损失曲线1
    plt.figure(figsize=(12, 8))
    # 从 episode=200 开始绘制
    start_episode = 200
    if len(episode_losses) > start_episode:
        # plt.plot(range(start_episode, len(episode_losses)), episode_losses[start_episode:], label='Loss')
        plt.plot(range(start_episode, len(episode_losses)), episode_losses[start_episode:])

    else:
        print("Not enough episodes to plot from episode 200.")
    plt.xlabel('Episode', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Average Loss', fontsize=12, fontname='Times New Roman')
    plt.title('Training Loss over Episodes (From Episode 200)', fontsize=12, fontname='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman'}, frameon=False)
    plt.grid(True)
    # 设置坐标轴刻度值的字体大小
    plt.tick_params(axis='both', which='major', labelsize=10, width=2)  # 设置刻度值字体大小
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontname('Times New Roman')  # 设置刻度值字体
        tick.set_fontweight('bold')  # 加粗刻度值字体


    # 设置纵轴范围
    if len(episode_losses) > start_episode:
        plt.ylim(0, max(episode_losses[start_episode:]) * 1.1)
    # 保存训练损失曲线图片
    loss_plot_path = os.path.join(plots_dir, "training_loss_from_episode_200.png")
    plt.savefig(loss_plot_path, bbox_inches='tight', dpi=600)
    plt.close()  # 关闭图像，避免内存泄漏

    # 绘制训练损失曲线2
    plt.figure(figsize=(12, 8))
    # plt.plot(range(len(episode_losses)), episode_losses, marker='o', label='Loss')
    plt.plot(range(len(episode_losses)), episode_losses)

    plt.xlabel('Episode', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Average Loss', fontsize=12, fontname='Times New Roman')
    plt.title('Training Loss over Episodes', fontsize=12, fontname='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman'}, frameon=False)
    plt.grid(True)

    # 设置坐标轴刻度值的字体大小
    plt.tick_params(axis='both', which='major', labelsize=10, width=2)  # 设置刻度值字体大小
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontname('Times New Roman')  # 设置刻度值字体
        tick.set_fontweight('bold')  # 加粗刻度值字体

    # 设置纵轴范围
    plt.ylim(0, max(episode_losses) * 1.1)

    # 保存训练损失曲线图片
    loss_plot_path = os.path.join(plots_dir, "training_loss.png")
    plt.savefig(loss_plot_path, bbox_inches='tight', dpi=600)
    plt.close()  # 关闭图像，避免内存泄漏

    # 绘制训练奖励曲线
    plt.figure(figsize=(10, 6))
    # plt.plot(range(1, episodes + 1), episode_rewards, label='Reward', color='orange')
    plt.plot(range(1, episodes + 1), episode_rewards, color='orange')

    plt.xlabel('Episode', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Total Reward', fontsize=12, fontname='Times New Roman')
    plt.title('Training Rewards over Episodes', fontsize=12, fontname='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman'}, frameon=False)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=10, width=2)  # 设置刻度值字体大小
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontname('Times New Roman')  # 设置刻度值字体
        tick.set_fontweight('bold')  # 加粗刻度值字体

    # 保存训练奖励曲线图片
    reward_plot_path = os.path.join(plots_dir, "training_rewards.png")
    plt.savefig(reward_plot_path, bbox_inches='tight',dpi=600)
    plt.close()  # 关闭图像，避免内存泄漏

    # 保存最终模型
    model_save_path = os.path.join(project_root, "drl\\weights", "dqn_gas_extraction_model_final_r1.pth")
    torch.save(agent.model.state_dict(), model_save_path)
    print("模型保存成功了")

if __name__ == "__main__":
    train()