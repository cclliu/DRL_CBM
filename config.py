# config.py
import yaml
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        # 这里可以动态获取配置文件路径，例如通过环境变量或者命令行参数
        config_path = '.\\config.yaml'
        with open(config_path, 'r', encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.data
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

# 初始化单例实例
config = Config()