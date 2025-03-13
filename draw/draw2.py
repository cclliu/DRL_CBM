import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_action_distribution_from_records(episode, save_dir):
    """
    从保存的动作记录中读取数据并生成动作分布的三维图
    """
    # 定义井名列表
    well_names = [
        "TS-010", "TS46-02", "TS46-03", "TS47-04", "TS47-05", "TS56-01", "TS56-02",
        "TS56-03", "TS56-04", "TS56-08", "TS5603D1", "TS57-01", "TS57-02", "TS57-03",
        "TS57-07", "TS57-09", "TS58-01"
    ]

    # 读取动作记录
    action_records_path = os.path.join(save_dir, f"action_records_episode_{episode}.txt")
    with open(action_records_path, "r") as f:
        action_records = [line.strip().split(": ")[1] for line in f.readlines()]

    # 只取倒数 54 行数据
    if len(action_records) > 54:
        action_records = action_records[-54:]

    # 将动作记录转换为数值
    episode_actions = []
    for actions in action_records:
        # 去除多余的字符（如 '[' 和 ']'），并将字符串转换为列表
        actions_cleaned = actions.replace("[", "").replace("]", "").replace(" ", "")
        actions_list = actions_cleaned.split(",")
        # 将字符串转换为整数
        episode_actions.append([int(action) for action in actions_list])
    episode_actions = np.array(episode_actions)

    # 创建三维图
    fig = plt.figure(figsize=(20, 8))  # 增加画布宽度
    ax = fig.add_subplot(111, projection='3d')

    # 确保时间步为正数
    time_steps = np.arange(len(episode_actions))  # 时间步从 0 开始
    if np.any(time_steps < 0):
        raise ValueError("Time steps contain negative values!")

    time_steps = time_steps[::-1]  # 将时间步顺序反转

    # 绘制每个井的动作分布
    for i in range(17):
        ax.scatter(time_steps, [i] * len(episode_actions), episode_actions[:, i], label=well_names[i])  # 使用井名作为标签

    ax.invert_xaxis()


    # 设置 z 轴的刻度和标签
    ax.set_zticks([0, 1, 2])
    ax.set_zticklabels(['↓', '→', '↑'],fontsize=26, fontname='Times New Roman',fontweight='bold')  # 使用箭头符号表示动作

    # 设置坐标轴标签
    ax.set_xlabel('Time Steps', fontsize=18, labelpad=10, fontname='Times New Roman',fontweight='bold')
    ax.set_ylabel('Well Index', fontsize=18, labelpad=10, fontname='Times New Roman',fontweight='bold')
    ax.set_zlabel('Action', fontsize=18, labelpad=3, fontname='Times New Roman',fontweight='bold')
    # ax.set_title(f'Action Distribution over Time Steps (Episode {episode})', fontsize=14, pad=20,
    #              fontname='Times New Roman')

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

    # 设置 x 和 y 轴的刻度标签样式
    plt.tick_params(axis='x', which='major', labelsize=14, width=1.5)
    plt.tick_params(axis='y', which='major', labelsize=14, width=1.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname('Times New Roman')
        # tick.set_fontweight('bold')

    # 单独设置 z 轴的刻度标签样式
    for tick in ax.get_zticklabels():
        tick.set_fontname('Times New Roman')
        tick.set_fontweight('bold')

    # 保存图片
    plot_path = os.path.join(save_dir, f"action_distribution_episode_{episode}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)  # 保存时调整边界，提高分辨率
    plt.close(fig)  # 关闭图像，避免内存泄漏

# 示例调用
# save_dir = r'H:\surrogate_CMB文字材料\论文撰写\过程图片\强化学习\r2_2000\records'
save_dir = r'E:\论文图\2025.1.15数据\r1_2000\过程记录'

# 遍历所有 episode 文件
for episode in range(1, 2001, 100):  # 从 1 到 2000，每隔 100 个 episode
    try:
        plot_action_distribution_from_records(episode, save_dir)
        print(f"成功生成 episode {episode} 的图")
    except Exception as e:
        print(f"生成 episode {episode} 的图时出错: {e}")