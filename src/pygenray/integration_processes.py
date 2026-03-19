"""
Ray equations and the methods needed to solve them. Implemented using JAX for
JIT-compilation and automatic differentiation support. The ray equations are
derived from the Hamiltonian formulation for ray theory [Colosi2016]_.

.. math::
    y = \\left [ t, z, p \\right ]^T

where :math:`t` is the travel time, :math:`z` is the depth, and :math:`p` is
the ray parameter :math:`(\\frac{sin(\\theta)}{c})`, and range, :math:`x` is
the independant variable.

.. math :: \\frac{dT}{dx} = \\frac{1}{c\\sqrt{1-c^2 \\ p_z^2}} \\\\
    :label: ray1
.. math :: \\frac{dz}{dx} = \\frac{c \\ p_z}{ \\sqrt{1-c^2 \\ p_z^2}} \\\\
    :label: ray2
.. math :: \\frac{dp_z}{dx} = -\\frac{1}{c^2}\\frac{1}{\\sqrt{1-c^2 \\ p_z^2}}\\frac{\\partial c}{\\partial z} \\\\
    :label: ray3

References
----------
.. [Colosi2016] Colosi, J. A. (2016). Sound Propagation through the Stochastic Ocean, Cambridge University Press, 443 pages.

"""
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def bilinear_interp(x, y, x_grid, y_grid, values):
    """
    Perform bilinear interpolation on a 2D grid.

    Parameters
    ----------
    x : float
        The x-coordinate at which to interpolate.
    y : float
        The y-coordinate at which to interpolate.
    x_grid : array_like
        1-D array of x-coordinates of the grid points, sorted ascending.
    y_grid : array_like
        1-D array of y-coordinates of the grid points, sorted ascending.
    values : array_like
        2-D array of shape (len(x_grid), len(y_grid)) containing the values.

    Returns
    -------
    float
        The interpolated value at point (x, y).
    """
    i = jnp.clip(jnp.searchsorted(x_grid, x, side='right') - 1, 0, x_grid.shape[0] - 2)
    j = jnp.clip(jnp.searchsorted(y_grid, y, side='right') - 1, 0, y_grid.shape[0] - 2)
    wx = (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
    wy = (y - y_grid[j]) / (y_grid[j + 1] - y_grid[j])
    return ((1 - wx) * (1 - wy) * values[i, j]
            + wx * (1 - wy) * values[i + 1, j]
            + (1 - wx) * wy * values[i, j + 1]
            + wx * wy * values[i + 1, j + 1])


@jax.jit
def linear_interp(x, xin, yin):
    """
    Perform linear interpolation on a 1D grid.

    Parameters
    ----------
    x : float
        The x-coordinate at which to interpolate.
    xin : array_like
        1-D array of x-coordinates, sorted ascending.
    yin : array_like
        1-D array of values at each grid point.

    Returns
    -------
    float
        The interpolated value at point x.
    """
    i = jnp.clip(jnp.searchsorted(xin, x, side='right') - 1, 0, xin.shape[0] - 2)
    w = (x - xin[i]) / (xin[i + 1] - xin[i])
    return (1 - w) * yin[i] + w * yin[i + 1]


def derivsrd(x, y, args):
    """
    Compute the differential equations for ray propagation.

    diffrax ODE signature: vector_field(t, y, args) -> dy.

    Parameters
    ----------
    x : float
        horizontal range (meters)
    y : jnp.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    args : tuple
        (cin, cpin, rin, zin, depths, depth_ranges)

    Returns
    -------
    dydx : jnp.array (3,)
        derivative of ray variables with respect to horizontal range
    """
    cin, cpin, rin, zin, depths, depth_ranges = args
    c = bilinear_interp(x, y[1], rin, zin, cin)
    cp = bilinear_interp(x, y[1], rin, zin, cpin)
    fact = 1.0 / jnp.sqrt(1.0 - c ** 2 * y[2] ** 2)
    return jnp.array([fact / c, c * y[2] * fact, -fact * cp / c ** 2])


def surface_bounce(x, y, cin, cpin, rin, zin, depths, depth_ranges):
    """Surface event condition: negative below surface, zero at z=0."""
    return -y[1]


def bottom_bounce(x, y, cin, cpin, rin, zin, depths, depth_ranges):
    """Bottom event condition: negative above bottom, zero at bottom depth."""
    return y[1] - linear_interp(x, depth_ranges, depths)


def vertical_ray(x, y, cin, cpin, rin, zin, depths, depth_ranges):
    """Vertical ray event condition: negative unless |θ| >= 89.999 degrees."""
    c = bilinear_interp(x, y[1], rin, zin, cin)
    return jnp.abs(jnp.degrees(jnp.arcsin(jnp.clip(y[2] * c, -1., 1.)))) - (90. - 1e-3)


def ray_bounding_box_event(x, y, cin, cpin, rin, zin, depths, depth_ranges):
    """Bounding box event: positive if ray is outside the sound speed grid."""
    z = y[1]
    outside = (z > zin[-1]) | (z < zin[0]) | (x < rin[0]) | (x > rin[-1])
    return jnp.where(outside, 1.0, -1.0)


def event_cond(t, y, args, **kw):
    """Combined event condition for surface, bottom, and vertical-ray termination."""
    cin_, cpin_, rin_, zin_, depths_, dr_ = args
    c = bilinear_interp(t, y[1], rin_, zin_, cin_)
    theta = jnp.degrees(jnp.arcsin(jnp.clip(y[2] * c, -1., 1.)))
    return jnp.max(jnp.array([
        -y[1],
        y[1] - linear_interp(t, dr_, depths_),
        jnp.abs(theta) - (90.0 - 1e-3),
    ]))


def ray_angle(x, y, cin, rin, zin):
    """
    Calculate angle of ray for specific ray state.

    Parameters
    ----------
    x : float
        horizontal range (meters)
    y : array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    cin : array (m,n)
        2D array of sound speed values
    rin : array (m,)
        range coordinate for c arrays
    zin : array (n,)
        depth coordinate for c arrays

    Returns
    -------
    theta : float
        angle of ray (degrees)
    c : float
        sound speed at ray state (m/s)
    """
    c = bilinear_interp(x, y[1], rin, zin, cin)
    theta = jnp.degrees(jnp.arcsin(jnp.clip(y[2] * c, -1., 1.)))
    return float(theta), float(c)


__all__ = [
    'derivsrd',
    'bottom_bounce',
    'surface_bounce',
    'ray_bounding_box_event',
    'ray_angle',
    'bilinear_interp',
    'linear_interp',
    'vertical_ray',
    'event_cond',
]
