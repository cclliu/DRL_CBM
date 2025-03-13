import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import config

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
start_column=config.get("start-column")
end_column=config.get("end-column")
def create_xy_file(dataset, n_past, output_rule='current'):
    input_data, target_data = [], []
    for i in range(n_past, len(dataset)):
        input_data.append(dataset[i - n_past:i, :start_column])
        if output_rule == 'next':
            target_data.append(dataset[i, start_column:end_column])
        elif output_rule == 'current':
            # target_data.append(dataset[i - 1, 6:11])
            target_data.append(dataset[i - 1, start_column:end_column])
        elif output_rule == 'range':
            target_data.append(dataset[i - n_past:i, start_column:end_column])
        else:
            raise ValueError("Invalid output_rule specified. Choose from 'current', 'next', 'range'.")
    return np.array(input_data), np.array(target_data)

def create_xy_folder(folder_path, n_past, output_rule='current'):
    all_data = []

    # 定义输入列和输出列的索引
    # input_columns = list(range(0, 6))  # 第一列到第六列
    # output_columns = list(range(6, 11))  # 第七列到第十一列
    input_columns = list(range(0, start_column))  # 第一列到第18列
    output_columns = list(range(start_column, end_column))  # 第19列到第35列

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            all_data.append(df)

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 初始化 MinMaxScaler
    scaler_input = MinMaxScaler(feature_range=(0, 1))
    scaler_output = MinMaxScaler(feature_range=(0, 1))

    # 仅对输入列进行缩放
    combined_df.iloc[:, input_columns] = scaler_input.fit_transform(combined_df.iloc[:, input_columns]).astype('float64')
    # 仅对输出列进行缩放
    combined_df.iloc[:, output_columns] = scaler_output.fit_transform(combined_df.iloc[:, output_columns]).astype('float64')

    # 保存缩放器
    joblib.dump(scaler_input, os.path.join(project_root, "scaler\\scaler_input.pkl"))
    joblib.dump(scaler_output, os.path.join(project_root, "scaler\\scaler_output.pkl"))

    # 将缩放后的数据转换为 NumPy 数组
    combined_df_scaled = combined_df.to_numpy()

    # 分割数据并处理每个文件的数据
    start_idx = 0
    all_input_data = []
    all_target_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            end_idx = start_idx + len(df)
            df_scaled = combined_df_scaled[start_idx:end_idx]
            start_idx = end_idx

            input_data, target_data = create_xy_file(df_scaled, n_past=n_past, output_rule=output_rule)
            all_input_data.append(input_data)
            all_target_data.append(target_data)

    all_input_data = np.concatenate(all_input_data, axis=0)
    all_target_data = np.concatenate(all_target_data, axis=0)
    print("Input data shape:", all_input_data.shape)
    print("Target data shape:", all_target_data.shape)
    return all_input_data, all_target_data

if __name__ == '__main__':
    create_xy_folder(os.path.join(project_root, r"data\true\test"), 4)


