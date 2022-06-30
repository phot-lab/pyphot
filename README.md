## PyPhot

A Python library for the simulation of optical fiber transmission, characterized by the ability of running on both CPU and GPU to gain powerful acceleration.

### User guide

First, you need to install necessary dependencies, open the CLI and enter this command:

```shell
pip3 install -r requirements.txt
```

The usage of this library is very simple, just create a python script under the root directory and use the following code to import PyPhot

```python
import phot
```

Then you can call functions of PyPhot by `phot.init_globals()`, `phot.LaserSource()`, etc. There is an **example.py** under the root directory showing how to use our API. Please read it carefully.

### Generate requirements.txt

**requirements.txt** is useful for installation of related dependencies, you can generate it by command line tool **pipreqs**, turn to the root dir and enter this command to do it:

```shell
pipreqs . --mode gt --force
```

