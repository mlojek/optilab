from cec2017.functions import f1
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = [i for i in range(1000)]
    y = [f1([[xval, xval]]) for xval in x]
    plt.plot(x, y)
    plt.show()
