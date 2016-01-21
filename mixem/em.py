import numpy as np


def em(data, distributions, initial_weights, initial_parameters, max_iterations=100, tol=1e-15, tol_iters=10, progress_callback=None):

    n_distr = len(distributions)
    n_data = data.shape[0]

    weight = np.array(initial_weights)
    param = initial_parameters

    last_ll = np.zeros((tol_iters, ))
    resp = np.empty((n_data, n_distr))
    density = np.empty((n_data, n_distr))

    iteration = 0
    while True:
        # E-step #######

        # compute responsibilities
        for d in range(n_distr):
            density[:, d] = distributions[d].density(data, param[d])

        # normalize responsibilities of distributions so they sum up to one for example
        resp = weight[np.newaxis, :] * density
        resp /= np.sum(resp, axis=1)[:, np.newaxis]

        log_likelihood = np.sum(resp * np.log(density))

        # M-step #######
        for d in range(n_distr):
            param[d] = distributions[d].estimate_parameters(data, resp[:, d])

        weight = np.mean(resp, axis=0)

        if progress_callback:
            progress_callback(iteration, weight, param, log_likelihood)

        # Convergence check #######
        if iteration >= tol_iters and (last_ll[-1] - log_likelihood) / last_ll[-1] <= tol:
            last_ll[0] = log_likelihood
            break

        if iteration >= max_iterations:
            break

        # store value of current iteration in last_ll[0]
        # and shift older values to the right
        last_ll[1:] = last_ll[:-1]
        last_ll[0] = log_likelihood

        iteration += 1

    return weight, param, last_ll[0]
