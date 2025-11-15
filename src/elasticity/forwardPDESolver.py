from dolfin import *

import matplotlib.pyplot as plt

# Create the mesh, spaces and boundaries
Lx = 1.0
Ly = 0.3
nx = 100
ny = 30
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)
V = VectorFunctionSpace(mesh, "CG", 1)

# Boundary markers: 1 = left (clamped), 2 = right (traction)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx)
right = RightBoundary()
right.mark(boundaries, 2)

# Measure for Neumann BCs (uses boundary tags)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Dirichlet BC: clamped left boundary
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
bc_left = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)

# Trial/test (also reused)
u = TrialFunction(V)
v = TestFunction(V)

def lame_parameters(E, nu):
    """Compute Lamé parameters from Young's modulus E and Poisson's ratio nu."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu

def epsilon(u):
    return sym(grad(u))
def sigma(u, lam, mu):
    return lam * tr(epsilon(u)) * Identity(2) + 2.0 * mu * epsilon(u)

def solve_elasticity(E, nu, traction=5e6):
    """
    Solve 2D linear elasticity on a fixed plate mesh for given (E, nu).

    Parameters
    ----------
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    traction : float
        Horizontal traction on the right boundary (Pa).

    Returns
    -------
    u_sol : Function
        Displacement field in VectorFunctionSpace V.
    """
    lam_val, mu_val = lame_parameters(E, nu)
    lam = Constant(lam_val)
    mu = Constant(mu_val)
    T = Constant((traction, 0.0))

    a = inner(sigma(u, lam, mu), epsilon(v)) * dx
    L = inner(T, v) * ds(2)

    u_sol = Function(V)
    solve(a == L, u_sol, bc_left)

    return u_sol

def demo_forward():
    """
    Run a forward solve for visualization and plot u_x and u_y over the plate.
    """
    E = 150e9
    nu = 0.28
    traction = 5e6
    u_sol = solve_elasticity(E, nu, traction=traction)

    # Split into components
    u_x, u_y = u_sol.split(deepcopy=True)

    plt.figure(figsize=(8,4))
    p = plot(u_x)  # FEniCS' own plotting helper
    plt.colorbar(p, label=r"$u_x(x,y)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title( fr"$X$-displacement field ($E={E/1e9:.1f}$ GPa, $\nu={nu:.2f}$, $T={traction/1e6:.1f}$ MPa)")
    plt.tight_layout()

    plt.figure(figsize=(8,4))
    p = plot(u_y)  # FEniCS' own plotting helper
    plt.colorbar(p, label=r"$u_y(x,y)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title( fr"$Y$-displacement field ($E={E/1e9:.1f}$ GPa, $\nu={nu:.2f}$, $T={traction/1e6:.1f}$ MPa)")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    demo_forward()