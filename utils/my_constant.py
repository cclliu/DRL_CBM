import torch

from config import config
from surrogate_model.cnn_lstm import CNNLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_cnn_lstm_surrogate_model(pth_path: str):
    # 模拟已有的CNN-LSTM模型

    input_size: int = config.get("surrogate_model.input_size")
    hidden_size: int = config.get("surrogate_model.hidden_size")
    num_layers: int = config.get("surrogate_model.num_layers")
    output_size: int = config.get("surrogate_model.output_size")
    # 创建一个简易的3层神经网络，输入为6，隐藏层为30，输出层为5
    model: CNNLSTM = CNNLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             output_size=output_size)

    # todo 模拟加载的模型参数，pth文件是怎么来的？哪里训练好的？肯定有地方执行了torch.save()函数
    # torch.load()函数用来加载状态字典，而model.load_state_dict()函数用于将字段应用到模型上
    pth_data = torch.load(pth_path)
    model.load_state_dict(pth_data)
    # 加载张量数据到GPU
    model.to(device)
    # 设定为评价模式
    model.eval()
    print("模型参数加载成功")
    return model

