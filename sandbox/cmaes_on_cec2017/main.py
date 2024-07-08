import cma
import matplotlib.pyplot as plt
import numpy as np
from cec2017.functions import f1


def run_cmaes_on_cec(cec_function: callable, dims: int):
    '''
    TODO
    '''
    x_start = np.random.normal(size=(1, dims))
    print(x_start)
    x_start = [0 for _ in range(dims)]
    xopt, es = cma.fmin2(
        lambda x: cec_function([x]),
        x_start,
        1.0
    )
    # TODO how to plot the results????


if __name__ == '__main__':
    run_cmaes_on_cec(f1, 2)
