import os
import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from surrogate_model.preprocess import create_xy_folder
from utils import date_util
from utils.my_constant import create_cnn_lstm_surrogate_model

# 设置项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_input = joblib.load(os.path.join(project_root, r"scaler\scaler_input.pkl"))
scaler_output = joblib.load(os.path.join(project_root, r"scaler\scaler_output.pkl"))

def evaluate_model(model, data_loader, criterion, scaler_output, dataset_type="Test"):
    """
    评估模型在数据集上的表现，并输出 RMSE、MAE 和 R^2 分数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    actuals = []
    predictions = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            actuals.extend(targets.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

            # 打印前几个样本的输入、目标和预测值
            if i == 0:  # 只打印第一个batch
                print(f"Sample Input ({dataset_type}): {inputs[0].cpu().numpy()}")
                print(f"Sample Target ({dataset_type}): {targets[0].cpu().numpy()}")
                print(f"Sample Prediction ({dataset_type}): {outputs[0].cpu().numpy()}")

    avg_loss = total_loss / len(data_loader.dataset)
    print(f'{dataset_type} Loss (MSE): {avg_loss:.4f}')

    # 转换为 NumPy 数组用于计算指标
    actuals = np.array(actuals)
    predictions = np.array(predictions)

    # 反归一化
    actuals_rescaled = scaler_output.inverse_transform(actuals)
    predictions_rescaled = scaler_output.inverse_transform(predictions)

    # 计算评价指标
    rmse = np.sqrt(mean_squared_error(actuals_rescaled, predictions_rescaled))
    mae = mean_absolute_error(actuals_rescaled, predictions_rescaled)
    r2 = r2_score(actuals_rescaled, predictions_rescaled)

    print(f'{dataset_type} RMSE: {rmse:.4f}')
    print(f'{dataset_type} MAE: {mae:.4f}')
    print(f'{dataset_type} R^2 Score: {r2:.4f}')

    return actuals_rescaled, predictions_rescaled

def save_plot(save_dir, plot_name):
    """
    保存当前图像到指定目录，并生成唯一文件名。
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = date_util.get_today_str()
    save_path = os.path.join(save_dir, f"{plot_name}_{timestamp}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 提高分辨率
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_45_degree(actuals_train, predictions_train, actuals_test, predictions_test, save_dir=None, well_names=None):
    """
    绘制45度线图，用于比较预测值和实际值。
    """
    if well_names is None:
        well_names = [f"Output {i + 1}" for i in range(actuals_train.shape[1])]

    for i in range(actuals_train.shape[1]):
        plt.figure(figsize=(6, 6))
        plt.scatter(actuals_train[:, i], predictions_train[:, i], color='blue', alpha=0.6, label='Train')
        plt.scatter(actuals_test[:, i], predictions_test[:, i], color='green', alpha=0.6, label='Test')
        plt.plot([actuals_train[:, i].min(), actuals_train[:, i].max()],
                 [actuals_train[:, i].min(), actuals_train[:, i].max()],
                 color='red', linestyle='--', linewidth=2, label='45 Degree Line')
        plt.xlabel('Actual Values', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Predicted Values', fontsize=12, fontname='Times New Roman')
        plt.title(f'{well_names[i]}', fontsize=14, fontname='Times New Roman')
        plt.legend(fontsize=12, prop={'family': 'Times New Roman'})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_dir:
            save_plot(save_dir, f"45_degree_line_{well_names[i]}")

    plt.figure(figsize=(10, 10))
    for i in range(actuals_train.shape[1]):
        plt.scatter(actuals_train[:, i], predictions_train[:, i], alpha=0.6, label=f'{well_names[i]} (Train)')
        plt.scatter(actuals_test[:, i], predictions_test[:, i], alpha=0.6, label=f'{well_names[i]} (Test)')
    plt.plot([actuals_train.min(), actuals_train.max()],
             [actuals_train.min(), actuals_train.max()],
             color='red', linestyle='--', linewidth=2, label='45 Degree Line')
    plt.xlabel('Actual Values', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Predicted Values', fontsize=12, fontname='Times New Roman')
    plt.title('45 Degree Line for All Wells', fontsize=14, fontname='Times New Roman')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'family': 'Times New Roman'})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_dir:
        save_plot(save_dir, "45_degree_line_all_wells")

def calculate_metrics(actuals, predictions, well_names, save_dir):
    """
    计算每口井的R²、RMSE、MAPE 和 SMAPE，并保存到文件中。
    """
    metrics = []
    for i in range(actuals.shape[1]):
        actual = actuals[:, i]
        pred = predictions[:, i]
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))

        mask = actual != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
        else:
            mape = np.nan

        smape = np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred))) * 100

        metrics.append([well_names[i], r2, rmse, mape, smape])

    metrics_df = pd.DataFrame(metrics, columns=["Well Name", "R^2", "RMSE", "MAPE", "SMAPE"])
    print("Metrics for each well:")
    print(metrics_df)

    # 保存到CSV文件
    metrics_path = os.path.join(save_dir, "well_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    return metrics_df

def plot_relative_error_boxplot(actuals, predictions, save_dir=None, well_names=None):
    """
    绘制每口井的绝对相对误差箱型图，并计算中位数和平均绝对相对误差。
    """
    absolute_relative_errors = []
    for i in range(actuals.shape[1]):
        actual = actuals[:, i]
        pred = predictions[:, i]
        mask = actual != 0
        if np.sum(mask) > 0:
            errors = np.abs((actual[mask] - pred[mask]) / actual[mask]) * 10
            absolute_relative_errors.append(errors)
        else:
            absolute_relative_errors.append(np.array([]))

    median_errors = [np.median(errors) if len(errors) > 0 else np.nan for errors in absolute_relative_errors]
    mean_errors = [np.mean(errors) if len(errors) > 0 else np.nan for errors in absolute_relative_errors]

    for i, well_name in enumerate(well_names):
        print(f"{well_name} - Median Absolute Relative Error: {median_errors[i]:.2f}%, Mean Absolute Relative Error: {mean_errors[i]:.2f}%")

    plt.figure(figsize=(12, 6))
    boxplot = plt.boxplot(absolute_relative_errors, labels=well_names, patch_artist=True, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'green', 'markersize': 8}, showfliers=False)  # 不显示异常点

    for box in boxplot['boxes']:
        box.set_facecolor('lightblue')
    for median in boxplot['medians']:
        median.set_color('orange')
        median.set_linewidth(2)
    for whisker in boxplot['whiskers']:
        whisker.set_color('black')
    for cap in boxplot['caps']:
        cap.set_color('black')

    plt.xticks(rotation=45, fontname='Times New Roman',fontsize=14)
    plt.xlabel('Well Name', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Absolute Relative Error (%)', fontsize=14, fontname='Times New Roman')
    plt.title('Absolute Relative Error Boxplot for Each Well', fontsize=16, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_dir:
        save_plot(save_dir, "absolute_relative_error_boxplot")

    return median_errors, mean_errors

def load_data(folder_path, n_past):
    """
    加载数据并转换为张量。
    """
    input_data, target_data = create_xy_folder(folder_path, n_past=n_past)
    X_tensor = torch.tensor(input_data, dtype=torch.float32)
    y_tensor = torch.tensor(target_data, dtype=torch.float32)
    return X_tensor, y_tensor

def main():
    # 加载模型
    timestr = date_util.get_today_str()
    model_path = os.path.join(current_dir, "weights", f"cnn_lstm_model_parameters_{timestr}.pth")
    model = create_cnn_lstm_surrogate_model(model_path)

    # 打印模型结构
    print("Model Architecture:")
    print(model)

    # 定义参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_folder_path = os.path.join(project_root, r"data\true\train")
    test_folder_path = os.path.join(project_root, r"data\true\test")
    n_past = 4
    batch_size = 32

    # 加载数据
    X_train, y_train = load_data(train_folder_path, n_past)
    X_test, y_test = load_data(test_folder_path, n_past)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义损失函数
    criterion = torch.nn.MSELoss()

    # 评估模型
    actuals_train, predictions_train = evaluate_model(model, train_loader, criterion, scaler_output, dataset_type="Train")
    actuals_test, predictions_test = evaluate_model(model, test_loader, criterion, scaler_output, dataset_type="Test")

    # 设置保存图像的目录
    save_dir = os.path.join(project_root, "figures", f"test_results_{date_util.get_today_str()}")
    os.makedirs(save_dir, exist_ok=True)

    # 井的名称
    well_names = ["TS-010", "TS46-02", "TS46-03", "TS47-04", "TS47-05", "TS56-01", "TS56-02", "TS56-03", "TS56-04",
                  "TS56-08", "TS5603D1", "TS57-01", "TS57-02", "TS57-03", "TS57-07", "TS57-09", "TS58-01"]

    # 计算并保存每口井的指标
    metrics_df = calculate_metrics(actuals_test, predictions_test, well_names, save_dir)  # 传递 save_dir 参数

    # 绘制绝对相对误差箱型图
    plot_relative_error_boxplot(actuals_test, predictions_test, save_dir=save_dir, well_names=well_names)
    # plot_relative_error_boxplot(actuals_train, predictions_train, save_dir=save_dir, well_names=well_names)


    # 绘制45度线图
    # plot_45_degree(actuals_train, predictions_train, actuals_test, predictions_test, save_dir=save_dir, well_names=well_names)

if __name__ == '__main__':
    main()