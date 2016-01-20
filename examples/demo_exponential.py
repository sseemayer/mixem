#!/usr/bin/env python

import numpy as np
import mixem
import mixem.distribution


def generate_data():
    dist_params = [1, 10]
    weights = [0.4, 0.6]

    n_data = 10000
    data = np.zeros((n_data))
    for i in range(n_data):
        dpi = np.random.choice(range(len(dist_params)), p=weights)
        dp = dist_params[dpi]
        data[i] = np.random.exponential(scale=1 / dp)

    return data


def progress(iter, weights, params, ll):
    print("{0:4d}: ll={1:.5e} w={2:.2f}     l0={3:.4e} l1={4:.4e}".format(iter, ll, weights[0], params[0], params[1]))


def recover(data):

    init_params = np.random.choice(data, size=2)
    init_weights = [0.2, 0.8]

    weight, param, ll = mixem.em(data, [mixem.distribution.ExponentialDistribution] * 2, init_weights, init_params, progress_callback=progress)

    print(weight, param, ll)


if __name__ == '__main__':
    data = generate_data()
    recover(data)
