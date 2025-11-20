import numpy as np

from elasticity import solve_elasticity, create_sensor_positions, observe

E_true = 1.0
nu_true = 0.28

# Solve the pde with the true paramters and observe in the sensor locations.
u_solution = solve_elasticity(E_true, nu_true)
sensors = create_sensor_positions()
u_obs = observe(u_solution, sensors) # shape (N_sensors, 2)

# Add Gaussian noise
rng = np.random.RandomState()
N_data_points = 10
sigma_x = 1.e-1
sigma_y = 1.e-2
measurement_noise_x = sigma_x * rng.normal(0.0, 1.0, size=(N_data_points, sensors.shape[0]))
measurement_noise_y = sigma_y * rng.normal(0.0, 1.0, size=(N_data_points, sensors.shape[0]))
observations = u_obs[np.newaxis,:,:] + np.concatenate((measurement_noise_x[:,:,np.newaxis], measurement_noise_y[:,:,np.newaxis]), axis=2)

# Store
np.savez_compressed(f'../data/DIC_observations_N={N_data_points}.npz', observations=observations, sensors=sensors, noise_sigma_x=sigma_x, noise_sigma_y=sigma_y)