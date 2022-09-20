## Operations notes

### Generate requirements.txt

**requirements.txt** is useful for installation of related dependencies, you can generate it by command line tool **
pipreqs**, enter the root directory and execute this command:

```shell
pipreqs . --mode gt --force --ignore ./test
```

### Generate distribution files

In the same directory with the **pyproject.toml**

```shell
python3 -m build
```

### Upload package to PyPI

```shell
python3 -m twine upload dist/*
```

