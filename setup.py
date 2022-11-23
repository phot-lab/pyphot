import os
from setuptools import setup


# Utility function to read the README file.
def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name="phot",
    version="0.2.3",
    author="Chunyu Li",
    author_email="cyli0212@gmail.com",
    description=("A Python library for the simulation of optical fiber transmission."),
    long_description_content_type='text/markdown',
    long_description=read('README.md'),
    url="https://github.com/phot-lab/pyphot",
    package_dir={"phot": "phot"},
    packages=['phot'],
    python_requires=">=3.7",
    install_requires=[
        'matplotlib>=3.5.2',
        'numba>=0.55.2',
        'numpy>=1.22.4',
        'scikit_commpy>=0.7.0',
        'scipy>=1.8.1'
    ]
)
