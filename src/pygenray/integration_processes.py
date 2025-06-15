"""
Ray equations and the methods needed to solve them. These functions are the primary bottle neck for the numerical integration and are optimized for speed using Numba just in time compilation. The ray equations are derived from the Hamiltonian formulation for ray theory [Colosi2016]_.

.. math::
    y = \\left [ t, z, p \\right ]^T

where :math:`t` is the travel time, :math:`z` is the depth, and :math:`p` is the ray parameter :math:`(\\frac{sin(\\theta)}{c})`, and range, :math:`x` is the independant variable.

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
import numba
import numpy as np

@numba.njit(fastmath=True, cache=True)
def derivsrd(
        x : float,
        y : np.array,
        cin : np.array,
        cpin : np.array,
        rin : np.array,
        zin : np.array,
        depths: np.array,
        depth_ranges : np.array,
    ) -> np.array:
    '''
    Compute the differential equations for ray propagation. The ray equations are derived from the Hamiltonian formulation for ray theory [Colosi2016]_, which consist of three coupled ODEs with range as the independant varibale, given by equations :eq:`ray1`, :eq:`ray2`, and :eq:`ray3`.

    .. math::
        y = \\left [ t, z, p \\right ]^T

    where :math:`t` is the travel time, :math:`z` is the depth, and :math:`p` is the ray parameter :math:`(\\frac{sin(\\theta)}{c})`, and range, :math:`x` is the independant variable.

    .. math :: \\frac{dT}{dx} = \\frac{1}{c\\sqrt{1-c^2 \\ p_z^2}} \\\\
        :label: ray1
    .. math :: \\frac{dz}{dx} = \\frac{c \\ p_z}{ \\sqrt{1-c^2 \\ p_z^2}} \\\\
        :label: ray2
    .. math :: \\frac{dp_z}{dx} = -\\frac{1}{c^2}\\frac{1}{\\sqrt{1-c^2 \\ p_z^2}}\\frac{\\partial c}{\\partial z} \\\\
        :label: ray3

    Parameters
    ----------
    x : float
        horizontal range (meters)
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    cin : np.array (m,n)
        2D array of sound speed values
    cpin : np.array (m,n)
        2D array of dc/dz
    rin : np.array (m,)
        range coordinate for c arrays
    zin : np.array (n,)
        depth coordinate for c arrays
    depths : np.array (k,)
        array of bathymetry values
    depth_ranges : np.array(k,)
        array of bathymetry value ranges. Does not have to match rin grid.

    Returns
    -------
    dydx : np.array (3,)
        derivative of ray variables with respect to horizontal range, [dT/dx, dz/dx, dp/dx]

    References
    ----------
    .. [Colosi2016] Colosi, J. A. (2016). Sound Propagation through the Stochastic Ocean, Cambridge University Press, 443 pages.
    '''
    
    #unpack ray variables
    z=y[1] # current depth
    pz=y[2] # current ray parameter

    #interpolate sound speed and its derivative at current depth and range
    c = bilinear_interp(x,z,rin,zin,cin)
    cp = bilinear_interp(x,z,rin,zin,cpin)

    # calculate derivatives
    fact=1/np.sqrt(1-(c**2)*(pz**2))
    dydx = np.array([
        fact/c,
        c*pz*fact,
        -fact*cp/(c**2)
    ])

    return dydx

@numba.njit(fastmath=True, cache=True)
def bilinear_interp(x, y, x_grid, y_grid, values):
    """
    Perform bilinear interpolation on a 2D grid.

    Fast, purely functional bilinear interpolation for scattered points on a 
    regular 2D grid using Numba JIT compilation for performance.

    Parameters
    ----------
    x : float
        The x-coordinate at which to interpolate.
    y : float
        The y-coordinate at which to interpolate.
    x_grid : array_like
        1-D array of x-coordinates of the grid points, must be sorted in 
        ascending order.
    y_grid : array_like
        1-D array of y-coordinates of the grid points, must be sorted in
        ascending order.
    values : array_like
        2-D array of shape (len(x_grid), len(y_grid)) containing the values
        at each grid point.

    Returns
    -------
    float
        The interpolated value at point (x, y).

    Notes
    -----
    This function uses bilinear interpolation, which linearly interpolates
    first in one dimension, then in the other. The interpolation is performed
    using the four nearest grid points surrounding the query point.

    If the query point lies outside the grid bounds, it is clamped to the
    nearest edge of the grid before interpolation.

    The function is compiled with Numba's JIT compiler for improved performance.

    Examples
    --------
    >>> import numpy as np
    >>> x_grid = np.array([0.0, 1.0, 2.0])
    >>> y_grid = np.array([0.0, 1.0, 2.0])
    >>> values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> result = bilinear_interp(0.5, 0.5, x_grid, y_grid, values)
    >>> print(result)  # Should be 3.0
    """

    # Find grid indices
    i = np.searchsorted(x_grid, x) - 1
    j = np.searchsorted(y_grid, y) - 1
    
    # Clamp to grid bounds
    i = max(0, min(i, len(x_grid) - 2))
    j = max(0, min(j, len(y_grid) - 2))
    
    # Bilinear weights
    wx = (x - x_grid[i]) / (x_grid[i+1] - x_grid[i])
    wy = (y - y_grid[j]) / (y_grid[j+1] - y_grid[j])
    
    # Interpolate
    v00 = values[i, j]
    v10 = values[i+1, j] 
    v01 = values[i, j+1]
    v11 = values[i+1, j+1]
    
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11

@numba.njit(fastmath=True, cache=True)
def linear_interp(x, xin, yin):
    """
    Perform linear interpolation on a 1D grid.

    Fast, purely functional linear interpolation for scattered points on a 
    regular 1D grid using Numba JIT compilation for performance.

    Parameters
    ----------
    x_interp : float
        The x-coordinate at which to interpolate.
    xin : array_like
        1-D array of x-coordinates of the grid points, must be sorted in 
        ascending order.
    yin : array_like
        1-D array of shape (len(x_grid),) containing the values
        at each grid point.

    Returns
    -------
    y_interp : float
        The interpolated value at point x.

    Notes
    -----
    This function uses linear interpolation between the two nearest grid points
    surrounding the query point.

    If the query point lies outside the grid bounds, it is clamped to the
    nearest edge of the grid before interpolation.

    The function is compiled with Numba's JIT compiler for improved performance.

    Examples
    --------
    >>> import numpy as np
    >>> x_grid = np.array([0.0, 1.0, 2.0])
    >>> values = np.array([1.0, 4.0, 7.0])
    >>> result = linear_interp(0.5, x_grid, values)
    >>> print(result)  # Should be 2.5
    """
    
    # Find grid index
    i = np.searchsorted(xin, x) - 1
    
    # Clamp to grid bounds
    i = max(0, min(i, len(xin) - 2))
    
    # Linear weight
    w = (x - xin[i]) / (xin[i+1] - xin[i])
    
    # Interpolate
    v0 = yin[i]
    v1 = yin[i+1]
    
    y_interp = (1-w)*v0 + w*v1

    return y_interp

@numba.njit(fastmath=True, cache=True)
def bottom_bounce(x,y,cin, cpin, rin, zin, depths, depth_ranges):
    """
    Parameters
    ----------
    x : float
        horizontal range (meters)
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    cin : np.array (m,n)
        2D array of sound speed values
    cpin : np.array (m,n)
        2D array of dc/dz
    rin : np.array (m,)
        range coordinate for c arrays
    zin : np.array (n,)
        depth coordinate for c arrays
    depth : np.array(m,)
        array of depths (meters), should be 1D with shape (m,)
    depth_ranges : np.array(m,)
        array of depth ranges (meters), should be 1D with shape (m,)

    Returns
    -------
    value : np.array (2,)
        bounce event values for surface and bottom
    isterminal : np.array (2,)
        boolean array indicating if the event should stop the integration
    direction : np.array (2,)
        direction of the event, 1 for increasing, -1 for decreasing
    """

    water_depth = linear_interp(x, depth_ranges, depths)
    ray_depth = y[1]

    # in original code, bottom bounce is enforced within 2 meters
    # I'm removing this for now, which might cause numerical issues
    #if np.abs(val2) < 2:
    #    val2=0

    # crosses zero for bottom reflection
    bottom_term = ray_depth-water_depth

    return bottom_term  # bounce event at surface and bottom

@numba.njit(fastmath=True, cache=True)
def surface_bounce(x,y,cin, cpin, rin, zin, depths, depth_ranges):
    """
    Parameters
    ----------
    x : float
        horizontal range (meters)
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    cin : np.array (m,n)
        2D array of sound speed values
    cpin : np.array (m,n)
        2D array of dc/dz
    rin : np.array (m,)
        range coordinate for c arrays
    zin : np.array (n,)
        depth coordinate for c arrays
    depth : np.array(m,)
        array of depths (meters), should be 1D with shape (m,)
    depth_ranges : np.array(m,)
        array of depth ranges (meters), should be 1D with shape (m,)

    Returns
    -------
    value : np.array (2,)
        bounce event values for surface and bottom
    isterminal : np.array (2,)
        boolean array indicating if the event should stop the integration
    direction : np.array (2,)
        direction of the event, 1 for increasing, -1 for decreasing
    """
    ray_depth = y[1]
    return ray_depth 

@numba.njit(fastmath=True, cache=True)
def ray_bounding_box_event(x,y,cin, cpin, rin, zin, depths, depth_ranges):
    '''
    Ray Bounding Box Event - trigger when ray position goes outside of bounding box. Bounding box is defined as the box where sound speed is defined.

    Returns
    -------
    bbox : bool
        True if ray is outside bounding box, False otherwise
    '''

    z = y[1]

    bbox = (z > zin[-1]) | (z < zin[0]) | (x < rin[0]) | (x > rin[-1])
    if bbox:
        print('bbox event triggered', z, zin[-1], zin[0], x, rin[0], rin[-1])
    return bbox

@numba.njit(fastmath=True, cache=True)
def ray_angle(
        x : float,
        y : np.array,
        cin : np.array,
        rin : np.array,
        zin : np.array
):
    """
    calculate angle of ray for specific ray state
    
    Parameters
    ----------
    x : float
        horizontal range (meters)
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    cin : np.array (m,n)
        2D array of sound speed values
    rin : np.array (m,)
        range coordinate for c arrays
    zin : np.array (n,)
        depth coordinate for c arrays

    Returns
    -------
    theta : float
        angle of ray (degrees)
    c : float
        sound speed at ray state (m/s)
    """

    c = bilinear_interp(x, y[1], rin, zin, cin)
    theta = np.degrees(np.asin(y[2] * c))
    return theta,c


__all__ = [
    'derivsrd',
    'bottom_bounce',
    'surface_bounce',
    'ray_bounding_box_event',
    'ray_angle',
    'bilinear_interp',
    'linear_interp',
]

