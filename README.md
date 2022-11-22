# PyPhot

A Python library for the simulation of optical fiber transmission, characterized by the ability of running on both CPU and GPU to gain powerful acceleration.

## Quick start

```shell
# Install PyPhot library
pip3 install phot

# Run example
python3 example.py
```

The Python file **example.py** under the root directory showing how to use our API (for PyPhotEngine). Please read it carefully.


## Notes for developers

### Generate requirements.txt

**requirements.txt** is useful for installation of related dependencies, you can generate it by command line tool `pipreqs`, enter the **root directory** and execute this command to generate **requirements.txt**:

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


## Deprecated phot module

The deprecated phot module is in `deprecated` folder. It is just reserved for developers. If you are user, just ignore it.

