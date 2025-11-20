import numpy as np
import matplotlib.pyplot as plt

from elasticity import logPriorDensity, loglikelihood
from typing import Dict, Optional

def plotPosterior(N_data_points : int, 
                  points : Dict = {},
                  trajectory : Optional[np.ndarray] = None):
    # Load the data from file
    data = np.load(f'../data/DIC_observations_N={N_data_points}.npz')
    observations = data["observations"]
    sensors = data["sensors"]
    noise_sigma_x = data["noise_sigma_x"]
    noise_sigma_y = data["noise_sigma_y"]

    # Split into x/y components for your loglikelihood
    u_x_ref = observations[:, :, 0]  # (N_data, N_sensors)
    u_y_ref = observations[:, :, 1]

    # Make a grid of (nu, log10E) values
    n_grid_values = 101
    nu_values = np.linspace(-0.2, 0.4, n_grid_values)
    log10E_values = np.linspace(-1, 2, n_grid_values)
    grid_nu_values, grid_log10E_values = np.meshgrid(nu_values, log10E_values)

    log_prior_values = np.zeros_like(grid_nu_values)
    log_likelihood_values = np.zeros_like(grid_nu_values)
    for i in range(n_grid_values):
        nu = nu_values[i]
        for j in range(n_grid_values):
            print((i,j))
            log10E = log10E_values[j]

            log_prior_values[i,j] = logPriorDensity(nu, log10E)
            log_likelihood_values[i,j] = loglikelihood(nu, log10E, u_x_ref, u_y_ref, sensors, noise_sigma_x, noise_sigma_y)
    prior_values = np.exp(log_prior_values)
    posterior_values = np.exp(log_likelihood_values - np.max(log_likelihood_values)) * prior_values
    is_inf = np.isinf(prior_values)
    prior_values[is_inf] = np.nan
    posterior_values[is_inf] = np.nan

    colors = {"MAP" : 'red', "true" : 'black'}
    labels = {"MAP" : "MAP", "true" : "Ground Truth"}

    # make a 3D plot of the posterior distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_nu_values, grid_log10E_values, posterior_values.T, cmap='viridis')
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\log10 E$")
    ax.set_zlabel(r"$p(\nu, E | data)$")
    ax.set_title('Posterior Density')

    fig = plt.figure()
    cs = plt.contourf(grid_nu_values, grid_log10E_values, prior_values.T, levels=40)
    plt.colorbar(cs, label=r"$p(\nu, \log_{10} E)$")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\log_{10} E$")
    plt.title('Prior Density')

    plt.figure()
    cs = plt.contourf(grid_nu_values, grid_log10E_values, posterior_values.T, levels=40)
    plt.colorbar(cs, label=r"$p(\nu,\log_{10}E \mid \text{data})$")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\log_{10} E$")
    for p in points.keys():
        plt.scatter(points[p][0], points[p][1], label=labels[p], marker='x', color=colors[p])
    if trajectory is not None:
        plt.plot(trajectory[:,0], trajectory[:,1], color='red', marker='.')
    if len(points.keys()) > 0:
        plt.legend()
    plt.show()

if __name__ == '__main__':
    points = {"true": np.array([0.28, 0.0])}
    plotPosterior(10, points)