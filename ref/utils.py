import sys
from pathlib import Path

path = Path(__file__)
parent_directory = path.parent.parent.absolute()  # 获取父目录

sys.path.append(str(parent_directory))  # 将父目录加入 Python 路径
