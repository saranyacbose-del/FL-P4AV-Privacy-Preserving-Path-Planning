import numpy as np

def add_noise(w, b, sigma=0.1):
    return w + np.random.normal(0, sigma, w.shape), b + np.random.normal(0, sigma)
