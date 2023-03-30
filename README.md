# PyPhot

一个用于光纤仿真运算的 Python 库。目前已经实现的组件包括：

1. 随机二进制生成器
2. QAM调制器
3. RRC脉冲整形
4. DAC的量化噪声
5. 激光器产生的相位噪声
6. 高斯白噪声
7. 双偏振的光纤传输信道
8. 收发端激光器造成的频偏
9. 接收机造成的I/Q失衡
10. ADC的量化噪声
11. IQ正交化补偿
12. 频偏估计和补偿
13. 帧同步
14. 自适应均衡
15. BPS相位恢复
16. 分析器画星座图和眼图
17. 计算误码率和 Q 影响因子

此外，我们还实现了多种方式的加速效果：

1. 光纤传输部分采用了 GPU 加速
2. 发射端和接收端组件采用了 JIT 加速

## 运行代码

```shell
# 安装相关依赖
pip install -r requirements.txt

# 运行代码示例
python example.py
```

主目录下的 Python 文件 [example.py](example.py) 展示了 PyPhot  所有组件 API 的使用方式。

## TOML 配置文件

TOML 配置文件是用于前端定义组件和计算引擎实现自动调度的，都位于 [toml](toml/) 目录下。

## 开发者文档

[开发者文档](docs/developer-guide.md) 提供了对于开发者的指导。

## 组件清单

组件分为两种：[函数](docs/components-list/function-list.md) 和 [类](docs/components-list/class-list.md)。
