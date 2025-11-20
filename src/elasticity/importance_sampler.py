import numpy as np

from elasticity import loglikelihood, logPosterior, logPriorDensity, samplePriorDensity

def importance_sampling(u_x_ref : np.ndarray,
                        u_y_ref : np.ndarray,
                        sensors : np.ndarray,
                        noise_sigma_x : float,
                        noise_sigma_y : float):
    # Sample the prior distribution
    N = 10_000
    prior_samples = samplePriorDensity(N)

    # Give each of the particles a weight given by the likelihood
    log_weights = np.zeros(N)
    for n in range(N):
        log_weight = loglikelihood(prior_samples[n,0], prior_samples[n,1], u_x_ref, u_y_ref, sensors, noise_sigma_x, noise_sigma_y)
        log_weights[n] = log_weight

    # Subtract the maximum for more numerical stability, then exponentiate
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
