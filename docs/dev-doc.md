# Development document

## Clone the repo

Note the `--recursive` option which is needed for the `pybind11` submodule:

```shell
git clone --recursive https://github.com/phot-lab/pyphot.git
```

## Test the project locally

Under the root dir

```shell
pip install -e .
```

Then you can use the modules and functions of **PyPhot** simply by `import phot`.

## Generate requirements.txt

**requirements.txt** is useful for the installation of related dependencies, you can generate it by command line tool `pipreqs`, enter the root directory and execute this command:

```shell
pipreqs . --mode gt --force --ignore ./test
```

## Generate distribution files

In the same directory with the **setup.py**:

```shell
# Build both source and binary distribution files
python3 -m build
```

## Upload package to PyPI

Use this command to upload to PyPI, notice that you should ensure the **project version** is different with the exiasting one on PyPI, and you need to **clear** the `dist` directory first because the old version distribution file will remain.

```shell
python3 -m twine upload dist/*
```

## Legacy files

The legacy phot module is in the `legacy` folder. It stores old version of `phot` project and has the value of reference.
