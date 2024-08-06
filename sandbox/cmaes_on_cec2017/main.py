import cma
import matplotlib.pyplot as plt
import numpy as np
from cec2017.functions import all_functions
from cec2017.functions import *


def run_cmaes_on_cec(cec_function: callable, dims: int):
    '''
    TODO
    '''
    # 4 + 3sqrt(dim)
    # 4dim
    x_start = np.random.normal(size=(4*dims))
    print(x_start)
    # x_start = [0 for _ in range(dims)]
    # TODO ograniczenia kostkowe - odbijanie Lamarck'a
    xopt, es = cma.fmin2(
        lambda x: cec_function([x]),
        x_start,
        1.0
    )
    # cma.plot()
    # cma.s.figsave('cmaes_result.png')


if __name__ == '__main__':
    # for f in all_functions:
    run_cmaes_on_cec(f8, 10)
