## PyPhot

A Python library for the simulation of optical fiber transmission, characterized by the ability of running on both CPU and GPU to gain powerful acceleration.

### User guide

First, you need to install necessary dependencies, open the CLI and enter this command:

```shell
pip3 install -r requirements.txt
```

There is an **example.py** under the root directory showing how to use our API (for PyPhotEngine). Please read it carefully.

### Generate requirements.txt

**requirements.txt** is useful for installation of related dependencies, you can generate it by command line tool **pipreqs**, enter the root directory and execute this command:

```shell
pipreqs . --mode gt --force --ignore ./test
```

### Deprecated phot module

The deprecated phot module is in `deprecated` folder. It is just reserved for developers. If you are user, just ignore it.
