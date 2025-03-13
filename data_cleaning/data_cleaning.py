#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/7 10:34
# @Author  : Liu Chen
# @File    : data_cleaning.py
# @Software: PyCharm
import json

import pandas as pd
import os
import re


def sanitize_filename(name):
    """
    清理文件名，移除非法字符。
    """
    # 替换非法字符为下划线
    return re.sub(r'[\/:*?"<>|]', '_', name)


def split_excel_columns(input_file, output_dir):
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取所有 sheet 名称
    excel_data = pd.ExcelFile(input_file)
    sheet_names = excel_data.sheet_names

    for sheet_name in sheet_names:
        # 清理 sheet 名称并创建子文件夹
        sanitized_sheet_name = sanitize_filename(sheet_name)
        sheet_folder = os.path.join(output_dir, sanitized_sheet_name)
        if not os.path.exists(sheet_folder):
            os.makedirs(sheet_folder)

        # 读取每个 Sheet
        sheet_data = pd.read_excel(input_file, sheet_name=sheet_name)
        print(f"Processing Sheet: {sheet_name}")

        # 遍历每一列
        for col_name in sheet_data.columns:
            # 清理列名
            sanitized_col_name = sanitize_filename(str(col_name))

            # 将当前列保存为新的 Excel 文件
            output_file = os.path.join(sheet_folder, f"{sanitized_col_name}.xlsx")
            col_data = sheet_data[[col_name]]  # 获取单列数据并保留 DataFrame 格式
            col_data.to_excel(output_file, index=False, sheet_name=sanitized_col_name)
            print(f"Saved {sanitized_col_name} to {output_file}")


def process_folder(root_folder):
    """
    处理文件夹1：
    1. 删除指定文件；
    2. 合并同名文件，并按列拼接，添加文件来源信息。
    """
    # 合并文件的输出目录
    merged_folder = os.path.join(root_folder, "merged")
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # 存储文件名与文件路径的映射
    file_mapping = {}

    # 遍历所有子文件夹和文件
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(subdir, file)

            # 删除名为 "Unnamed_0.xlsx" 的文件
            if file == "Unnamed_0.xlsx":
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
                continue

            # 记录同名文件
            if file not in file_mapping:
                file_mapping[file] = []
            file_mapping[file].append(file_path)

    # 合并同名文件，按列拼接
    for file_name, file_paths in file_mapping.items():
        combined_data = []

        for file_path in file_paths:
            # 读取文件数据
            data = pd.read_excel(file_path)
            # 添加一行，值为上一级文件夹名
            folder_name = os.path.basename(os.path.dirname(file_path))
            header_row = pd.DataFrame([[folder_name] * data.shape[1]], columns=data.columns)
            combined_data.append(pd.concat([header_row, data], ignore_index=True))

        # 按列拼接所有文件的数据
        merged_data = pd.concat(combined_data, axis=1, ignore_index=False)

        # 保存到合并目录
        merged_file_path = os.path.join(merged_folder, file_name)
        merged_data.to_excel(merged_file_path, index=False)
        print(f"Saved merged file: {merged_file_path}")


def process_folders(folder1, folder2, output_folder):
    """
    处理文件夹1和文件夹2中的Excel文件：
    1. 合并同名文件并删除第一行；
    2. 添加时间列；
    3. 修改表头：时间步、排水量、产气量。
    """
    # 合并文件的输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹1和文件夹2中的所有文件
    files_folder1 = {file: os.path.join(folder1, file) for file in os.listdir(folder1) if file.endswith('.xlsx')}
    files_folder2 = {file: os.path.join(folder2, file) for file in os.listdir(folder2) if file.endswith('.xlsx')}

    # 遍历文件夹1和文件夹2中的同名文件
    for file_name in files_folder1:
        if file_name in files_folder2:
            file_path1 = files_folder1[file_name]
            file_path2 = files_folder2[file_name]

            # 读取Excel文件并删除第一行
            data1 = pd.read_excel(file_path1, header=None)  # 不读取列名
            data2 = pd.read_excel(file_path2, header=None)  # 不读取列名

            # 删除第一行
            data1 = data1.drop(index=0).reset_index(drop=True)
            data2 = data2.drop(index=0).reset_index(drop=True)

            # 合并数据（按列拼接）
            merged_data = pd.concat([data1, data2], axis=1, ignore_index=True)

            time_column = pd.Series(range(0, len(merged_data) + 1))  # 从1开始递增
            merged_data.insert(0, "时间步", time_column)  # 将时间列插入第0列

            # 修改表头
            columns = ["时间步"]  # 第一列为时间步
            merged_data.iloc[0, 0] = "time step"

            # 遍历第一行的每个元素，修改内容
            for idx in range(len(merged_data.columns)):
                # 修改第一行每个单元格的值
                if idx == 0:
                    pass
                elif idx < 18:  # 18列以内，添加排水量
                    merged_data.iloc[0, idx] = merged_data.iloc[0, idx] + "排水量"
                else:  # 19列及之后，添加产气量
                    merged_data.iloc[0, idx] = merged_data.iloc[0, idx] + "产气量"

            # 保存合并后的文件
            merged_file_path = os.path.join(output_folder, file_name)
            merged_file_path = merged_file_path.replace("xlsx", "csv")
            merged_data.to_csv(merged_file_path, index=False, header=False)

            print(f"Saved merged file: {merged_file_path}")


def split_csv(input_file, output_file1, output_file2):
    # 读取输入的 CSV 文件
    df = pd.read_csv(input_file)
    # 选取前 18 列
    df_first_18_cols = df.iloc[:, :18]
    # 选取 18 列之后的数据
    df_remaining_cols = df.iloc[:, 18:]
    # 将前 18 列数据保存到 csv1
    df_first_18_cols.to_csv(output_file1, index=False)
    # 将 18 列之后的数据保存到 csv2
    df_remaining_cols.to_csv(output_file2, index=False)




if __name__ == '__main__':

    # # 示例用法
    # input_file = "E:\\昊镪\\cbmdatacsv\\gas.xlsx"
    # output_dir = "E:\\昊镪\\cbmdatacsv\\2"
    # split_excel_columns(input_file, output_dir)

    # root_folder = "E:\\昊镪\\cbmdatacsv\\2"  # 替换为实际路径
    # process_folder(root_folder)

    # folder1 = "E:\\昊镪\\cbmdatacsv\\inputdata_water"
    # folder2 = "E:\\昊镪\\cbmdatacsv\\outputdata_gas"
    # output_folder = "E:\\昊镪\\cbmdatacsv\\end"  # 替换为输出文件夹路径
    # process_folders(folder1, folder2, output_folder)

    # 调用函数
    input_file = r'H:\surrogate_CMB\data\true\历史数据.csv'  # 替换为你的输入 CSV 文件的实际路径
    output_file1 = r'H:\surrogate_CMB\data\true\history_water.csv'  # 替换为第一个输出文件的实际路径
    output_file2 = r'H:\surrogate_CMB\data\true\history_gas.csv'  # 替换为第二个输出文件的实际路径
    split_csv(input_file, output_file1, output_file2)
