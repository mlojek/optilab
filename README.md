# Optilab
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Docker Pulls](https://img.shields.io/docker/pulls/mlojek/optilab?logo=Docker&label=Dockerhub%20pulls)
![Read the Docs](https://img.shields.io/readthedocs/optilab)

Optilab is a lightweight and flexible python framework for testing black-box optimization.

## Features
- ✅ Intuitive interface to quickly prototype and run optimizers and metamodels.
- 📚 High quality documentation.
- 📈 Objective functions, optimizers, plotting and data handling.
- ⋙ CLI functionality to easily summarize results of previous experiments.
- 🚀 Multiprocessing for faster computation.

## How to install
Optilab has been tested to work on python versions 3.11 and above. To install it from PyPI, run:
```
pip install optilab
```
You can also install from source by cloning this repo and running:
```
make install
```

## Try the demos
Learn how to use optilab and fit it to your needs with demo notebooks in `demo` directory.

## CLI tool
Optilab comes with a powerful CLI tool to easily summarize your experiments. It allows for plotting the results and performing statistical testing to check for statistical significance in optimization results.
```
usage: python -m optilab [-h] [--hide_plots] [--test_y] [--test_evals] [--entries ENTRIES [ENTRIES ...]] [--raw_values] [--hide_outliers] [--no_save] pickle_path

Optilab CLI utility.

positional arguments:
  pickle_path           Path to pickle file or directory with optimization runs.

options:
  -h, --help            show this help message and exit
  --hide_plots          Hide plots when running the script.
  --test_y              Perform Mann-Whitney U test on y values.
  --test_evals          Perform Mann-Whitney U test on eval values.
  --entries ENTRIES [ENTRIES ...]
                        Space separated list of indexes of entries to include in analysis.
  --raw_values          If specified, y values below tolerance are not substituted by tolerance value.
  --hide_outliers       If specified, outliers will not be shown in the box plot.
  --no_save             If specified, no artifacts will be saved.
```

## Docker
This project comes with a docker container. You can pull it from dockerhub:
```
docker pull mlojek/optilab
```
Or build it yourself:
```
make docker
```