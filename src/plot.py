import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import math


def plot_simple_array(x_list, save_as_filename):
    fig, axs = plt.subplots(1, 1)
    start, end = axs.get_xlim()
    axs.xaxis.set_ticks(np.arange(start, end, 10))
    axs.set_title(save_as_filename)
    for x in x_list:
        y = [y for y in range(len(x))]
        axs.plot(y, x)
    for x in x_list:
        y = [y for y in range(len(x))]
        p5, p4, a, b, c, d = np.polyfit(y, x, 5)
        predict_amount = min(10_000, math.floor((len(x) ** 0.6)))
        axs.plot(
            [y for y in range(len(x) + predict_amount)],
            [
                p5 * i**5 + p4 * i**4 + a * i**3 + b * i**2 + c * i + d
                for i in range(len(x) + predict_amount)
            ],
        )
    axs.grid()
    plt.savefig(save_as_filename)
    plt.close()
