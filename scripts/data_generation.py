import numpy as np

from elasticity import solve_elasticity, create_sensor_positions, observe

E_true = 150.0 * 1e9
nu_true = 0.28

# Solve the pde with the true paramters and observe in the sensor locations.
u_solution = solve_elasticity(E_true, nu_true)
sensors = create_sensor_positions()
u_obs = observe(u_solution, sensors) # shape (N_sensors, 2)

# Add Gaussian noise
rng = np.random.RandomState()
N_data_points = 1000
sigma = 1.e-6
measurement_noise = sigma * rng.normal(0.0, 1.0, size=(N_data_points, sensors.shape[0], 2))
observations = u_obs[np.newaxis,:,:] + measurement_noise

# Store
np.savez_compressed('../data/DIC_observations.npz', observations=observations)