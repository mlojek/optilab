# Optilab
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Docker Image Version](https://img.shields.io/docker/v/mlojek/optilab?logo=Docker&label=Docker%20image%20version)


Optilab is a lightweight and flexible python framework for testing black-box optimization.

## Features
- Intuitive interface to quickly prototype and run optimizers and metamodels.
- High quality documentation.
- Objective functions, optimizers, plotting and data handling.
- CLI functionality to easily summarize results of previous experiments.
- Multiprocessing for faster computation.

## How to run
Optilab has been tested to work on the latest python versions. To install it, just run `make install`.

## Try the demos
If you're not sure how to start using optilab, see some examples in `demo` directory.

## Docker
This project comes with a docker container. You can pull it from dockerhub:
```
docker pull mlojek/optilab
```
Or build it yourself:
```
make docker
```