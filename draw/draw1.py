import os
import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from surrogate_model.preprocess import create_xy_folder, create_xy_file
from utils import date_util
from utils.my_constant import create_cnn_lstm_surrogate_model

# 设置项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_input = joblib.load(os.path.join(project_root, r"scaler\scaler_input.pkl"))
scaler_output = joblib.load(os.path.join(project_root, r"scaler\scaler_output.pkl"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置字体为新罗马18号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

def plot_predictions_for_csv(model, csv_file_path, n_past):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 分离输入和输出列
    input_data = df.iloc[:, :18].values
    output_data = df.iloc[:, 18:35].values

    # 使用加载的缩放器进行缩放
    input_data_scaled = scaler_input.transform(input_data)
    output_data_scaled = scaler_output.transform(output_data)

    # 创建输入和输出数据
    dataset = np.hstack((input_data_scaled, output_data_scaled))
    input_data, target_data = create_xy_file(dataset, n_past=n_past, output_rule='current')

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(target_data, dtype=torch.float32).to(device)

    # 进行预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    # 反缩放预测值和实际值
    predictions_rescaled = scaler_output.inverse_transform(predictions)
    actuals_rescaled = scaler_output.inverse_transform(y_tensor.cpu().numpy())

    print(f"predictions={predictions_rescaled}")
    print(f"actuals={actuals_rescaled}")

    # 设置保存图像的目录
    save_dir = os.path.join(project_root, "figures", f"test_results_{date_util.get_today_str()}")
    os.makedirs(save_dir, exist_ok=True)

    # 井的名称
    well_names = ["TS-010", "TS46-02", "TS46-03", "TS47-04", "TS47-05", "TS56-01", "TS56-02", "TS56-03", "TS56-04",
                  "TS56-08", "TS5603D1", "TS57-01", "TS57-02", "TS57-03", "TS57-07", "TS57-09", "TS58-01"]

    # 绘图并保存为独立的图
    num_outputs = actuals_rescaled.shape[1]
    for i in range(num_outputs):
        plt.figure(figsize=(10, 6))  # 设置单个图的大小
        # plt.plot(actuals_rescaled[:, i], label=f'Actual Values {well_names[i]}', marker='o')
        # plt.plot(predictions_rescaled[:, i], label=f'Predicted Values {well_names[i]}', marker='x')
        plt.plot(actuals_rescaled[:, i], label=f'Actual Values {well_names[i]}',color='red', linestyle='-', linewidth=2, marker='o')
        plt.plot(predictions_rescaled[:, i], label=f'Predicted Values {well_names[i]}',color='blue', linestyle='--', linewidth=1, marker='x')

        plt.xlabel('Time Step', fontweight='bold')
        plt.ylabel(f'Gas Production Value {well_names[i]}', fontweight='bold')
        plt.xticks(fontsize=18, fontname='Times New Roman')
        plt.yticks(fontsize=18, fontname='Times New Roman')
        # plt.title(f'Predicted vs Actual Values for {well_names[i]}', fontsize=20, fontname='Times New Roman')
        plt.legend(fontsize=18)
        plt.tight_layout()

        # 保存图像，分辨率为300dpi
        save_path = os.path.join(save_dir,f"predictions_{well_names[i]}.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()  # 关闭当前图，避免内存泄漏


# 加载模型参数
timestr = date_util.get_today_str()
model_path = os.path.join(project_root, "surrogate_model//weights", f"cnn_lstm_model_parameters_{timestr}.pth")
model = create_cnn_lstm_surrogate_model(model_path)

model.to(device)
model.eval()  # 设置为评估模式，避免训练时的 dropout 或 batch normalization 行为
print("Model parameters have been loaded successfully.")

# 用一个流程的测试数据测试模型
criterion = torch.nn.MSELoss()
csv_file_path = r'H:\surrogate_CMB\data\true\train\A001_e1_v00001.csv'
# 或者使用指定的 CSV 文件展示预测结果
plot_predictions_for_csv(model, csv_file_path, n_past=4)