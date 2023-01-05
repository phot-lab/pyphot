# 开发者文档

## Clone 仓库

注意 `--recursive` 参数是必要的，因为我们的仓库依赖了其他的子模块。

```shell
git clone --recursive https://github.com/phot-lab/pyphot.git
```

## 本地安装运行

在主目录下：

```shell
pip install -e .
```

然后你就可以通过 `import phot` 来使用 **PyPhot** 的模块和函数。

## 生成 requirements.txt

你可以在主目录下通过以下命令生成 **requirements.txt** ，注意你需要拥有 `pipreqs` 命令行工具：

```shell
pipreqs . --mode gt --force --ignore ./test
```

## 生成 distribution 文件

在 **setup.py** 的相同目录下:

```shell
# 同时构建 source and binary distribution files
python3 -m build
```

## 上传 package 到 PyPI

使用以下命令上传到 PyPI，注意你需要确保项目版本和 PyPI上已存在的项目版本不一样，否则会产生冲突，同时你需要清除 `dist` 目录下已存在的旧版 distribution 文件。

```shell
python3 -m twine upload dist/*
```

## 旧版 PyPhot

旧版 PyPhot 在 `legacy` 目录下。
