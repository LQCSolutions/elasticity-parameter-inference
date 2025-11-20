import numpy as np

from elasticity import MAPEstimator

from plot_posterior import plotPosterior

N_data_points = 1000
data = np.load(f'../data/DIC_observations_N={N_data_points}.npz')
observations = data["observations"]
sensors = data["sensors"]
noise_sigma_x = data["noise_sigma_x"]
noise_sigma_y = data["noise_sigma_y"]

# Split into x/y components for your loglikelihood
u_x_ref = observations[:, :, 0]  # (N_data, N_sensors)
u_y_ref = observations[:, :, 1]

optimization_result = MAPEstimator(u_x_ref, u_y_ref, sensors, noise_sigma_x, noise_sigma_y)

# Print the result
if optimization_result.success:
    nu_opt = optimization_result.x[0]
    log10_E_opt = optimization_result.x[1]
    print(fr'MAP Estimate Found. $\nu={nu_opt}$ and $\log_{10}(E) = {log10_E_opt}$')
else:
    print('MAP Estimator did not converge.')

# Plot the map estimate and true paramters on the posterior distrbution 
plotPosterior(N_data_points, {'MAP' : optimization_result.x, "true" : np.array([0.28, 0.0])}, optimization_result.trajectory)