import numpy as np
from dolfin import Point, Function

from elasticity.forwardPDESolver import Lx, Ly, demo_forward

def create_sensor_positions(nx: int = 5,
                            ny: int = 2,
                            x_margin: float = 0.1,
                            y_margin: float = 0.03,
                            stagger: bool = True) -> np.ndarray:
    """
    Return an array of shape (N_SENSORS, 2) with (x, y) DIC positions
    where we 'measure' displacement. We take extra care so that these 
    sensor positions are not FEM grid points. Generate these positions
    only once!

    Parameters
    ----------
    nx, ny : int
        Number of points in x and y directions (before optional staggering).
    x_margin, y_margin : float
        Margins from the domain boundaries, to avoid edge effects.
    stagger : bool
        If True, shift every other row in x to avoid alignment with FEM nodes.

    Returns
    -------
    sensors : (N_sensors, 2) array
        Interior sensor coordinates.
    """
    # Interior box for DIC region
    x_min = x_margin
    x_max = Lx - x_margin
    y_min = y_margin
    y_max = Ly - y_margin

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)

    sensor_list = []

    for j, y in enumerate(ys):
        row_xs = xs.copy()

        # Optional staggering: shift every other row by half a step
        # to avoid direct overlap with a FEM mesh node. This makes 
        # the setup more interesting and realistic.
        if stagger and j % 2 == 1 and nx > 1:
            dx = (x_max - x_min) / (nx - 1)
            row_xs = row_xs + 0.5 * dx

            # Clip back into the interior in case of overshoot
            row_xs = np.clip(row_xs, x_min + 0.25 * dx, x_max - 0.25 * dx)

        for x in row_xs:
            sensor_list.append([x, y])

    return np.array(sensor_list)

def observe(u_sol : Function,
            sensors : np.ndarray) -> np.ndarray:
    """
    Evaluate the displacement field u_sol at given sensor positions.

    Parameters
    ----------
    u_sol : dolfin.Function
        Vector-valued displacement field (CG1 on the plate).
    sensors : ndarray, shape (N, 2)
        Sensor positions (x_i, y_i).

    Returns
    -------
    y : np.ndarray, shape (N,2)
        Observed displacement vector:
        [u_x(x1,y1), u_y(x1,y1); u_x(x2,y2), u_y(x2,y2); ...].
    """
    N = sensors.shape[0]
    y = np.zeros((N,2), dtype=float)

    for i, (x_i, y_i) in enumerate(sensors):
        ux, uy = u_sol(Point(float(x_i), float(y_i)))
        y[i,0] = ux
        y[i,1] = uy

    return y

if __name__ == '__main__':
    sensors = create_sensor_positions(stagger=False)
    demo_forward(sensors)