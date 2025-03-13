import torch
from torch import Tensor, nn


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
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        前向传播函数，定义数据在网络中的流动过程。

        参数:
        - state: 输入的状态张量

        返回:
        - 经过网络计算后的Q值张量，形状为(1, action_size)，表示对应各个动作的Q值估计
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)