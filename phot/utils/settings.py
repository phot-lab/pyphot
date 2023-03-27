_plot = True
_backend = "numpy"


def config(plot=True, backend="numpy"):
    """设置全局变量

    Args:
        plot (bool, optional): 设置是否开启全局画图，默认为 True。
        backend (str, optional): 设置运算后端，可以选择"numpy", "cupy", "torch"，
        注意使用"cupy"的话，需要额外安装cupy: "pip install cupy".
    """
    global _plot
    global _backend
    _plot = plot
    _backend = backend
