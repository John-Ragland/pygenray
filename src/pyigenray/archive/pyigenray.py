"""
pyigenray - python based implementation of underwater acoustic ray simulation
"""
import xarray as xr
import numpy as np
from numba import jit
from scipy.integrate import solve_ivp

def rayintrd_bathy(
        y0,
        dx : float,
        max_range : float,
        c : xr.DataArray,
        bathy : xr.DataArray,
        verbose : bool = False
): 
    '''
    integrate rays through range-dependant soundspeed and bathymetry

    Parameters
    ----------
    y0 : 
        Initial condition vector containing [initial travel time, initial depth, initial ray parameter]
    dx : float
        Horizontal step size for ray path calculations (meters)
    max_range : float
        Maximum horizontal range for ray tracing (meters)
    c : xr.DataArray
        xr.DataArray containing sound speed profile array (m/s), should either be a 1D array with dimension ['depth'] or a 2D array with dimensions ['depth', 'range']. coordinates for both dimensions should be in meters.
    D :
        Water depth parameter (though seems to be used only in event detection)
    bathy : xr.DataArray
        Array of depths, should be 1D with dimension ['range'] and coordinates in meters

    For now, c(r,z) and dc(r,z)/dz are stored in memory, and the ray angles and range steps are looped over. This will inevitably cause memory issues for multiprocessing, and will need to be changed to writing c and dc/dz to disk for async reads.

    Variable Notes
    --------------
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]

    '''

    # check that c and bathy have correct dimensions
    if bathy.ndim != 1:
        raise ValueError("bathy must be a 1D xr.DataArray")
    
    if c.ndim == 1:
        if c.dims[0] != 'depth':
            raise ValueError("1D c must have dimension ['depth']")
        raise ValueError("1D sound speed profile not supported yet, must use 2D sound speed with dimensions ['depth', 'range']")
    elif c.ndim == 2:
        if set(c.dims) != set(['depth', 'range']):
            raise ValueError("2D c must have dimensions 'depth' and 'range' (in any order)")
    else:
        raise ValueError("c must be a 1D or 2D xr.DataArray")

    if bathy.ndim == 1:
        if bathy.dims[0] != 'range':
            raise ValueError("bathy must have dimension ['range']")
    else:
        raise ValueError("bathy must be a 1D xr.DataArray")

    # create array of range, ray steps
    xx=np.arange(0,max_range, dx)

    # compute sound speed gradients (uses second order accurate central differences), see [here](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html#numpy.gradient)
    if verbose:
        print("Computing sound speed gradients...")
    cp = c.differentiate('depth')
    cs = xr.Dataset({'c': c, 'cp': cp})

    # Solve the ODE with events
    if verbose:
        print("Integrating ray...")
        
    sol = solve_ivp(
        lambda x, y: derivsrd(x, y, cs),
        [xx[0], xx[-1]],  # time span
        y0,
        method='RK45',
        t_eval=xx,  # evaluation points
        events=lambda x, y: bounce_events_rd(x, y, bathy),
        rtol=1.0e-9,
        atol=1.0e-9,
        callback=print_state,
    )
    return xx, sol

def interpolate_c(cs, z, x):
    '''
    interpolate sound speed and it's derivative at given depth and range
    
    Parameters
    ----------
    cs : xr.Dataset
        xr.Dataset with variables ['c', 'cp'] c and dc/dz with dimensions ['depth', 'range']
    z : float
        depth in meters
    x : float
        range in kilometers

    Returns
    -------
    c : float
        sound speed at depth and range
    cp : float
        dc/dz at depth and range
    '''

    # interpolate sound speed for range and depth value
    cs_interp = cs.interp({'depth':z, 'range':x}, method='linear')
    c = cs_interp['c'].values
    cp = cs_interp['cp'].values

    return c,cp

def derivsrd(x : float, y : np.array, cs : xr.Dataset) -> np.array:
    '''
    differential equations for ray propagation.

    Parameters
    ----------
    x : float
        horizontal range (kilometers)
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    cs : xr.Dataset
        xr.Dataset with variables ['c', 'cp'] c and dc/dz with dimensions ['depth', 'range']

    Returns
    -------
    dydx : np.array (3,)
        derivative of ray variables with respect to horizontal range, [dT/dx, dz/dx, dp/dx]
    '''
    
    #unpack ray variables
    z=y[1] # current depth

    #interpolate sound speed and its derivative at current depth and range
    cs_interp = cs.interp({'depth':z, 'range':x}, method='linear')
    c = float(cs_interp['c'])
    cp = float(cs_interp['cp'])

    dydx = derivsrd_float(y, c, cp)
    
    return dydx

@jit(nopython=True, cache=True)
def derivsrd_float(y, c, cp):
    '''
    differential equations for ray propagation. sound speed and sound speed derivate inputs are already interpolated and are passed as floats

    Parameters
    ----------
    x : float
        horizontal range (kilometers)
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    c : float
        sound speed at current depth and range
    cp : float
        dc/dz at current depth and range
    '''
    pz=y[2] # ray parameter

    fact=1/np.sqrt(1-c**2*pz**2)

    dydx = np.empty(3)
    dydx[0]=fact/c
    dydx[1]=c*pz*fact
    dydx[2]=-fact*cp/c**2

    return dydx

def bounce_events_rd(x,y,bathy):
    """
    [value,isterminal,direction]
    D is water depth assumed constant
    added real for conmplex rays

    Parameters
    ----------
    x : float
        horizontal range (kilometers)
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    bathy : xr.DataArray
        array of depths (meters), should be 1D with dimension ['range']

    Returns
    -------
    value : np.array (2,)
        bounce event values for surface and bottom
    isterminal : np.array (2,)
        boolean array indicating if the event should stop the integration
    direction : np.array (2,)
        direction of the event, 1 for increasing, -1 for decreasing
    """

    DD=bathy.interp({'range':x},method='linear').values

    val2=y[1]+DD
    if np.abs(val2) < 2:
        val2=0

    value = np.array([y[1], val2])  # bounce event at surface and bottom
    isterminal = np.array([1, 1])   # stop at interfaces
    direction  = np.array([1, -1])  # [local minimum, local maximum]

    return value, isterminal, direction

