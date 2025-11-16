import numpy as np
import scipy.optimize as opt

from elasticity import logPosterior

def MAPEstimator(u_x_ref : np.ndarray,
                 u_y_ref : np.ndarray,
                 sensors : np.ndarray,
                 noise_sigma_x : float,
                 noise_sigma_y : float,
                 verbose=True) -> opt.OptimizeResult:

    # Build the negative log posterior
    objective = lambda p : -logPosterior(p[0], p[1], u_x_ref, u_y_ref, sensors, noise_sigma_x, noise_sigma_y)

    # minimize using scipy's minimize. We have no explicit Jacobian.
    bounds = [(-0.5, 0.5), (-3.0, 4.0)]
    def callback(intermediate_result: opt.OptimizeResult):
        if verbose:
            print(intermediate_result.x, intermediate_result.fun, np.linalg.norm(intermediate_result.grad))
    #initial_guess = np.array([0.1, 0.5]) # (nu, log10_E)
    initial_guess = np.array([0.1, 2.0])
    optimization_result = opt.minimize(objective, initial_guess, bounds=bounds, method='trust-constr', callback=callback, options={"gtol" : 1e-12})

    return optimization_result