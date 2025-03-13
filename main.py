import os

from drl import agent_train
from surrogate_model import train as surrogate_train
from config import config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 执行代理的训练
if __name__ == '__main__':
    print(config.get("surrogate-model-param.input_size"))
    # agent_train.train()
    surrogate_train.train()