# 数据预处理
import os
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from config import config
from surrogate_model.cnn_lstm import CNNLSTM
from surrogate_model.preprocess import create_xy_folder
from utils import date_util

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, scheduler=None, writer=None):
    model.train()
    train_losses = []  # 存储训练集每个 epoch 的平均损失
    val_losses = []    # 存储验证集每个 epoch 的平均损失

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0 and writer:
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)

        # 计算训练集平均损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证集损失计算
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        if scheduler:
            scheduler.step()

        if writer:
            writer.add_scalar('Average Train Loss per Epoch', avg_train_loss, epoch + 1)
            writer.add_scalar('Average Val Loss per Epoch', avg_val_loss, epoch + 1)

    print("Training finished.")
    return train_losses, val_losses

def draw_loss_curve(train_losses, val_losses, save_path=os.path.join(project_root, "figure",
                                                                     f"training_loss_curve_{date_util.get_today_str()}.png")):
    plt.figure(figsize=(config.get("figure.width"), config.get("figure.height")))

    # 设置字体大小和样式
    # plt.rcParams.update({'font.size': 9, 'font.family': 'Times New Roman'})

    # 绘制训练损失和验证损失曲线，并显示数据点
    # plt.legend(fontsize=28,  # 增大字体大小
    #            prop={'family': 'Times New Roman', 'weight': 'bold'},  # 加粗字体
    #            loc='upper left',
    #            frameon=True,
    #            framealpha=0.8,
    #            facecolor='white',
    #            edgecolor='black',
    #            borderpad=0.8,
    #            labelspacing=0.8)

    plt.plot(train_losses, label='Training Loss', linestyle='-', linewidth=2, marker='o', markersize=3)
    plt.plot(val_losses, label='Validation Loss', linestyle='--', linewidth=2, marker='^', markersize=3)

    # 设置标题和标签
    # plt.title('Training and Validation Loss Curve', fontsize=14, fontweight='bold')  # 标题加粗
    plt.xlabel('Epoch', fontsize=18,fontname='Times New Roman', fontweight='bold')  # x轴标签加粗
    plt.ylabel('Loss', fontsize=18, fontname='Times New Roman',fontweight='bold')  # y轴标签加粗
    plt.xticks( fontsize=18, fontname='Times New Roman')
    plt.yticks( fontsize=18, fontname='Times New Roman')

    # 设置图例
    plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 18})

    # 设置网格线
    # plt.grid(True, linestyle=':', alpha=0.6)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()
def train():
    # 定义参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_path = os.path.join(project_root, r"data\true\train")
    n_past = 4
    batch_size = 32
    num_epochs = config.get("train-episodes")

    input_data, target_data = create_xy_folder(folder_path, n_past=n_past)
    X_tensor = torch.tensor(input_data, dtype=torch.float32)
    y_tensor = torch.tensor(target_data, dtype=torch.float32)

    # 数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_size = 18
    hidden_size = 30
    num_layers = 3
    output_size = 17
    model = CNNLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=150, gamma=0.1)

    # 创建 TensorBoard writer
    writer = SummaryWriter(f'logs')
    example_input = torch.randn(batch_size, n_past, input_size).to(device)
    writer.add_graph(model, example_input)

    # 训练模型
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, scheduler=scheduler, writer=writer)

    # 保存模型参数
    time_str = date_util.get_today_str()
    model_save_path = os.path.join(current_dir, "weights", f"cnn_lstm_model_parameters_{time_str}.pth")
    torch.save(model.state_dict(), model_save_path)

    # 绘制训练和验证损失曲线
    draw_loss_curve(train_losses, val_losses)