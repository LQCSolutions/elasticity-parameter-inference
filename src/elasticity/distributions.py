import numpy as np

from elasticity import solve_elasticity, observe

nu_min = -0.2
nu_max = 0.5
mu_log10_E = 0.5
sigma_log10_E = 0.5

def logPriorDensity(nu : float,
                    logE : float) -> float:
    # Uniform in nu on [nu_min, nu_max]
    if not (nu_min <= nu <= nu_max):
        return -np.inf
    
    log_nu_density = np.log(1.0 / (nu_max - nu_min))
    log_log10E_density = -(logE - mu_log10_E)**2 / (2.0 * sigma_log10_E**2)  - np.log( np.sqrt(2.0*np.pi * sigma_log10_E**2) )
    
    return log_nu_density + log_log10E_density

def samplePriorDensity(N : int):
    rng = np.random.RandomState()
    nu_samples = rng.uniform(nu_min, nu_max, N)
    log10E_samples = rng.normal(mu_log10_E, sigma_log10_E, N)
    return np.concatenate((nu_samples[:,np.newaxis], log10E_samples[:,np.newaxis]), axis=1)

def loglikelihood(nu : float,
                  log10E : float,
                  u_x_ref : np.ndarray, # Shape (N_data, N_sensors)
                  u_y_ref : np.ndarray, # Shape (N_data, N_sensors)
                  sensors : np.ndarray,
                  noise_sigma_x : float,
                  noise_sigma_y : float) -> float: 
    
    # Solve the elastostatic PDE for (nu, E) parameters
    E = 10.0**log10E
    u_sol = solve_elasticity(E, nu)

    # Downsample for observations
    u_sol_obs = observe(u_sol, sensors) # shape (N_sensors, 2)

    # Compute the log-likelihood (negative potential)
    ux_loss = (u_sol_obs[np.newaxis, :, 0] - u_x_ref)**2 / noise_sigma_x**2
    uy_loss = (u_sol_obs[np.newaxis, :, 1] - u_y_ref)**2 / noise_sigma_y**2
    sq_error = float(np.sum(ux_loss + uy_loss))

    # Return the negative energy
    return -0.5 * sq_error

def logPosterior(nu : float,
                 log10E : float,
                 u_x_ref : np.ndarray, # Shape (N_data, N_sensors)
                 u_y_ref : np.ndarray, # Shape (N_data, N_sensors)
                 sensors : np.ndarray,
                 noise_sigma_x : float,
                 noise_sigma_y : float) -> float:
    return loglikelihood(nu, log10E, u_x_ref, u_y_ref, sensors, noise_sigma_x, noise_sigma_y) + logPriorDensity(nu, log10E)

def posterior(nu : float,
              log10E : float,
              u_x_ref : np.ndarray, # Shape (N_data, N_sensors)
              u_y_ref : np.ndarray, # Shape (N_data, N_sensors)
              sensors : np.ndarray,
              noise_sigma_x : float,
              noise_sigma_y : float) -> float:
    return np.exp(logPosterior(nu, log10E, u_x_ref, u_y_ref, sensors, noise_sigma_x, noise_sigma_y))