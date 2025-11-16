import numpy as np
import matplotlib.pyplot as plt

from elasticity import posterior, logPosterior, logPriorDensity, loglikelihood

def plotPosterior():
    # Load the data from file
    data = np.load('../data/DIC_observations.npz')
    observations = data["observations"]
    sensors = data["sensors"]
    noise_sigma_x = data["noise_sigma_x"]
    noise_sigma_y = data["noise_sigma_y"]

    # Split into x/y components for your loglikelihood
    u_x_ref = observations[:, :, 0]  # (N_data, N_sensors)
    u_y_ref = observations[:, :, 1]

    # Make a grid of (nu, log10E) values
    n_grid_values = 101
    nu_values = np.linspace(-0.5, 0.45, n_grid_values)
    log10E_values = np.linspace(-1, 1, n_grid_values)
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

    # make a 3D plot of the posterior distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_nu_values, grid_log10E_values, posterior_values.T, cmap='viridis')
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\log10 E$")
    ax.set_zlabel(r"$p(\nu, E | data)$")
    ax.set_title('Posterior Density')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_nu_values, grid_log10E_values, prior_values.T, cmap='viridis')
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\log10 E$")
    ax.set_zlabel(r"$p(\nu, E)$")
    ax.set_title('Prior Density')
    plt.show()

if __name__ == '__main__':
    plotPosterior()