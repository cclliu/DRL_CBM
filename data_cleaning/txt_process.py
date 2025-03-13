import csv
import os

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(infile):
            if i < 55:  # 只读取前54行
                outfile.write(line)
            else:
                break

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            process_file(input_file, output_file)
            print(f"Processed {filename}")



def csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 读取CSV文件
        with open(txt_file, 'w', encoding='utf-8') as txtfile:
            for row in csv_reader:
                # 将每一行转换为字符串并写入TXT文件
                txtfile.write(', '.join(row) + '\n')

# 设置CSV文件和TXT文件的路径
# csv_file = r'H:\surrogate_CMB文字材料\论文撰写\过程图片\强化学习\r2_2000\排水1\history_gas.csv'  # 替换为你的CSV文件路径
# txt_file =  r'H:\surrogate_CMB文字材料\论文撰写\过程图片\强化学习\r2_2000\排水1\history_gas.txt'  # 替换为你的TXT文件路径

# 调用函数将CSV文件转换为TXT文件
# csv_to_txt(csv_file, txt_file)
# print(f"CSV文件已成功转换为TXT文件: {txt_file}")

# # 设置输入和输出文件夹路径
input_folder = r'H:\surrogate_CMB文字材料\论文撰写\过程图片\强化学习\r2_2000\产气'
output_folder = r'H:\surrogate_CMB文字材料\论文撰写\过程图片\强化学习\r2_2000\产气2'
#
# 处理文件夹中的所有txt文件
process_folder(input_folder, output_folder)