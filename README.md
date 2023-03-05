# PyPhot

一个用于光纤仿真运算的 Python 库。

## 运行代码

```shell
# 安装相关依赖
pip install -r requirements.txt

# 运行代码示例
python example.py
```

主目录下的 Python 文件 [example.py](example.py) 展示了如何使用 PyPhot 的 API，请仔细阅读。

## TOML 配置文件

目前已给出发射端的 TOML 配置文件，以供计算引擎调用，都位于 [toml](toml/) 目录下。发射端的调用顺序为：

```
gen_bits() -> qam_modulate() -> PulseShaper() -> PulseShaper.tx_shape() -> dac_noise() -> phase_noise() -> gaussian_noise()
```

## 开发者文档

[开发者文档](docs/developer-guide.md) 提供了对于开发者的指导。

## 组件清单

组件分为两种：[函数](docs/components-list/function-list.md) 和 [类](docs/components-list/class-list.md)。
