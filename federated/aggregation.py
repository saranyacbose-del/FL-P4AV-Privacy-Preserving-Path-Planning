import numpy as np

def fedavg(weights, biases):
    w_avg = np.mean(weights, axis=0)
    b_avg = np.mean(biases)
    return w_avg, b_avg
