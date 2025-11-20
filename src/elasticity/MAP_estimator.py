import numpy as np

from elasticity import logPosterior

from dataclasses import dataclass
from typing import Callable

@dataclass
class OptimizeResult:
    success : bool
    x : np.ndarray
    fun : float
    grad : np.ndarray
    iters : int
    trajectory : np.ndarray

class AdamOptimizer:
    def __init__(self, lr=1.e-3):
        super().__init__()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lr = lr
        self.eps = 1e-8

    def optimize(self,
                 objective : Callable[[np.ndarray], float],
                 grad_objective : Callable[[np.ndarray], np.ndarray],
                 p0 : np.ndarray,
                 tol : float,
                 bounds : np.ndarray | None,
                 max_iter : int = 1000) -> OptimizeResult:
        trajectory = [p0]
        m = np.zeros_like(p0)
        v = np.zeros_like(p0)

        p = np.copy(p0)
        for t in range(1, max_iter + 1):
            g = grad_objective(p)
            if np.linalg.norm(g) < tol:
                return OptimizeResult(True, p, objective(p), g, t, np.array(trajectory))
            print(p, np.linalg.norm(g))
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            p = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if bounds is not None:
                p = np.clip(p, bounds[:,0], bounds[:,1])

            trajectory.append(p)

        g = grad_objective(p)
        success = bool(np.linalg.norm(g) <= tol)
        return OptimizeResult(success, p, objective(p), g, max_iter, np.array(trajectory))

class GradientDescentOptimizer:
    def __init__(self, lr=1.e-3):
        super().__init__()
        self.lr = lr

    def optimize(self,
                 objective : Callable[[np.ndarray], float],
                 grad_objective : Callable[[np.ndarray], np.ndarray],
                 p0 : np.ndarray,
                 tol : float,
                 bounds : np.ndarray | None,
                 max_iter : int = 1000) -> OptimizeResult:
        trajectory = [p0]

        p = np.copy(p0)
        for t in range(1, max_iter + 1):
            g = grad_objective(p)
            if np.linalg.norm(g) < tol:
                return OptimizeResult(True, p, objective(p), g, t, np.array(trajectory))
            print(p, np.linalg.norm(g))

            p = p - self.lr * g
            if bounds is not None:
                p = np.clip(p, bounds[:,0], bounds[:,1])
            trajectory.append(p)

        g = grad_objective(p)
        success = bool(np.linalg.norm(g) <= tol)
        return OptimizeResult(success, p, objective(p), g, max_iter, np.array(trajectory))
    
def MAPEstimator(u_x_ref : np.ndarray,
                 u_y_ref : np.ndarray,
                 sensors : np.ndarray,
                 noise_sigma_x : float,
                 noise_sigma_y : float,
                 tol : float = 1e-4) -> OptimizeResult:

    # Build the negative log posterior
    objective = lambda p : -logPosterior(p[0], p[1], u_x_ref, u_y_ref, sensors, noise_sigma_x, noise_sigma_y)

    # Build the objective gradient from finite differences
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    h = 1.e-4
    grad_objective = lambda p: np.array([objective(p + e1 * h) - objective(p - e1 * h), objective(p + e2 * h) - objective(p - e2 * h)]) / (2.0 * h)

    # Run the Adam optimizer
    lr = 1.e-4
    #adam = AdamOptimizer(lr=lr)
    adam = GradientDescentOptimizer(lr=lr)
    initial_guess = np.array([0.1, 0.5]) # (nu, log10_E)
    bounds = np.array([[-0.2, 0.4], [-3.0, 4.0]])
    result = adam.optimize(objective, grad_objective, initial_guess, tol, bounds)
    return result