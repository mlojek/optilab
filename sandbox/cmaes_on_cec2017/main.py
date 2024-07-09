import cma
import matplotlib.pyplot as plt
import numpy as np
from cec2017.functions import all_functions


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
    # cma.plot()
    # cma.s.figsave('cmaes_result.png')


if __name__ == '__main__':
    for f in all_functions:
        run_cmaes_on_cec(f, 10)
