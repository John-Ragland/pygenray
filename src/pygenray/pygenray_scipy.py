import numba
import numpy as np
import scipy.integrate
import pygenray as pr
import ocean_acoustic_env as oaenv
import scipy
from multiprocessing import shared_memory
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

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
def derivsrd_sp(
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
    differential equations for ray propagation.

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
def bottom_bounce_sp(x,y,cin, cpin, rin, zin, depths, depth_ranges):
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
def surface_bounce_sp(x,y,cin, cpin, rin, zin, depths, depth_ranges):
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

def shoot_ray_segment(
        x0 : float,
        y0 : np.array,
        reciever_range : float,
        cin : np.array,
        cpin : np.array,
        rin : np.array,
        zin : np.array,
        depths : np.array,
        depth_ranges : np.array,
        x_eval : np.array = None,
        rtol = 1e-6,
        terminate_backwards : bool = True
):
    """
    Given an initial condition vector and initial range, integrate ray
    until integration bounds or event is triggered (such as surface or bottom bounce).

    Parameters
    ----------
    x0 : float
        initial ray, x position
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    reciever_range : float
        integration range end bound. starting point is x0
    cin : np.array (m,n)
        2D array of sound speed values
    cpin : np.array (m,n)
        2D array of dc/dz
    rin : np.array (m,)
        range coordinate for c arrays
    zin : np.array (n,)
        depth coordinate for c arrays
    depths : np.array(m,)
        array of depths (meters), should be 1D with shape (m,)
    depth_ranges : np.array(m,)
        array of depth ranges (meters), should be 1D with shape (m,)
    x_eval : np.array
        array of x values at which to evaluate the solution
        (optional, if not provided, will use default t_eval for :func:`scipy.integrate.solve_ivp`)
    rtol : float
        relative tolerance for the ODE solver, default is 1e-6
    terminate_backwards : bool
        whether to terminate ray integration when ray starts propagating backwards. Default is True
    """

    # set up surface and bottom bounce events
    surface_event = surface_bounce_sp
    surface_event.terminal = True
    #surface_event.direction = -1
    
    bottom_event = bottom_bounce_sp  
    bottom_event.terminal = True
    #bottom_event.direction = 0

    events = (
        surface_event,
        bottom_event,
    )
    
    sol = scipy.integrate.solve_ivp(
        derivsrd_sp,
        (x0,reciever_range),
        y0,
        args = (cin, cpin, rin*1000, zin, depths, depth_ranges),
        t_eval=x_eval,
        events = events,
        rtol = rtol
    )

    return sol

def shoot_ray(
        source_depth : float,
        source_range : float,
        launch_angles : np.array,
        reciever_range : float,
        x_eval : np.array,
        environment : oaenv.environment.OceanEnvironment2D,
        rtol = 1e-6,
        terminal_backwards : bool = True,
        n_processes : int = None,
):
    '''
    given arrays of source depth, range, and launch angle, integrate ray to reciever range. For depth (m,), range (n,), and angle (k,), number of ray runs will be m*n*k.

    Parameters
    ----------
    source_depth : np.array
        array of source depths (meters), should be 1D with shape (m,)
    source_range : np.array
        array of source ranges (meters), should be 1D with shape (n,)
    launch_angle : np.array
        array of source angles (degrees), should be 1D with shape (k,)
    reciever_range : float
        reciever range (meters)
    x_eval : np.array
        The range values to save the ray state at.s
    environment : pr.environment.OceanEnvironment
        OceanEnvironment object specifying sound speed and bathymetry.
    rtol : float
        relative tolerance for the ODE solver, default is 1e-6
    terminate_backwards : bool
        whether to terminate ray if it bounces backwards
    n_processes : int
        number of processes to use, Default of None (mp.cpu_count)

    Returns
    -------
    ray : np.array
        2D array of ray state at each x_eval point, shape (4, n_eval), where n_eval is the number of evaluation points
    n_bott : int
        number of bottom bounces
    n_surf : int
        number of surface bounces
    '''

    if n_processes == None:
        n_processes = mp.cpu_count()
    # set up initial conditions for ray variable

    ## unpack environment object
    cin = np.array(environment.sound_speed.values)
    cpin = np.array(environment.sound_speed.differentiate('depth').values)
    rin = np.array(environment.sound_speed.range.values)
    zin = np.array(environment.sound_speed.depth.values)
    depths = np.array(environment.bathymetry.values)
    depth_ranges = np.array(environment.bathymetry.range.values)
    
    # check that coordinates are monotonically increasing
    if not (np.all(np.diff(rin) >= 0)):
        raise Exception('Sound speed range coordinates must be monotonically increasing.')
    if not (np.all(np.diff(zin) >= 0)):
        raise Exception('Sound speed depth coordinates must be monotonically increasing.')
    if not (np.all(np.diff(depth_ranges) >= 0)):
        raise Exception('Bathymetry range coordinates must be monotonically increasing.')
    
    # Create Shared Arrays
    array_metadata, shms = _init_shared_memory(cin, cpin, rin ,zin, depths, depth_ranges, environment.bottom_angle, x_eval)

    ## calculate sound speed at source location
    c = pr.bilinear_interp(source_range, source_depth, rin, zin, cin)

    # initial ray variables, time=0
    y0s = [np.array([0, source_depth, np.sin(np.radians(launch_angle))/c]) for launch_angle in launch_angles]

    shoot_ray_part = partial(
        shoot_single_ray,
        source_range=source_range,
        reciever_range=reciever_range,
        array_metadata=array_metadata,
        rtol=rtol,
        terminate_backwards=terminal_backwards
    )
    
    with mp.Pool(n_processes) as pool:
        results = list(tqdm(pool.imap(shoot_ray_part, y0s), total=len(y0s), desc="Processing rays"))

    # close and unlink shared memory
    for var in shms:
        shms[var].unlink()
        shms[var].close()

    return results
    return ray, n_bott, n_surf

def shoot_single_ray(
        y0 : np.array,
        source_range : float,
        reciever_range : float,
        array_metadata : dict,
        rtol = 1e-6,
        terminate_backwards : bool = True,
):
    """
    Given an initial condition vector and initial range, integrate ray
    until integration bounds or event is triggered.

    Parameters
    ----------
    y0 : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    source_range : float
        initial ray, x position
    reciever_range : float
        integration range end bound. starting point is x0
    array_metedata : dict
        dictionary containing metadata of shared memory arrays specificing environment. Calculated with `pr._init_shared_memory()`.
            cin, cpin, rin, zin, depths, depth_ranges, bottom_angle, x_eval
    rtol : float
        relative tolerance for the ODE solver, default is 1e-6

    Returns
    -------
    full_ray : np.array
        2D array of ray state at each x_eval point, shape (4, n_eval), where n_eval is the number of evaluation points
    n_bottom : int
        number of bottom bounces
    n_surface : int
        number of surface bounces
    """

    # Access shared arrays
    shared_arrays, existing_shms = _unpack_shared_memory(array_metadata)

    cin = shared_arrays['cin']
    cpin = shared_arrays['cpin']
    rin = shared_arrays['rin']
    zin = shared_arrays['zin']
    depths = shared_arrays['depths']
    depth_ranges = shared_arrays['depth_ranges']
    bottom_angle = shared_arrays['bottom_angle']
    x_eval = shared_arrays['x_eval']

    # initialize loop parameters
    x_intermediate = source_range
    full_ray = np.concat((np.array([source_range]), y0)).copy()
    full_ray = np.expand_dims(full_ray, axis=1)
    sols = []
    n_surface = 0
    n_bottom = 0
    loop_count = 0

    # create cubic interpolation of bottom slope
    bottom_angle_interp = scipy.interpolate.interp1d(
        depth_ranges,
        bottom_angle,
        kind='cubic'
    )

    # set intermediate ray state to initial ray state
    y_intermediate = y0.copy()

    while x_intermediate < reciever_range:

        x_eval_filtered = x_eval[x_eval >= x_intermediate]

        sol = shoot_ray_segment(
            x_intermediate,
            y_intermediate,
            reciever_range,
            cin,
            cpin,
            rin,
            zin,
            depths,
            depth_ranges,
            x_eval_filtered,
            rtol=rtol,
            terminate_backwards=terminate_backwards
        )

        sols.append(sol)
        full_ray = np.append(full_ray, np.vstack((sol.t, sol.y)), axis=1)

        # if end of integration is reached, end loop
        if sol.message == 'The solver successfully reached the end of the integration interval.':
            break

        y_intermediate = sol.y[:,-1]

        # Check if bounce event occurred and update x_intermediate accordingly
        if len(sol.t_events[0]) > 0 or len(sol.t_events[1]) > 0:
            # An event occurred, use the event time as the new range
            if len(sol.t_events[0]) > 0:  # Surface event
                x_intermediate = sol.t_events[0][0]
            elif len(sol.t_events[1]) > 0:  # Bottom event  
                x_intermediate = sol.t_events[1][0]

        else:
            # No event, use the final integration point
            x_intermediate = sol.t[-1]

        # calculate ray angle and sound speed at ray state
        theta,c = _ray_angle(x_intermediate, y_intermediate, cin, rin, zin)
        
        # Surface Bounce
        if len(sol.t_events[0])==1:
            theta_bounce = -theta
            n_surface += 1

        # Bottom Bounce
        elif len(sol.t_events[1])==1:
            beta = bottom_angle_interp(x_intermediate)
            theta_bounce = -theta + 2*beta
            n_bottom += 1
            # TODO double check bottom reflection angle

        #else: # no bounce event, but also not end of integration
        #    loop_count += 1
        #    continue
        #    # TODO add event handling for errors
        
        # terminate if ray bounces backwards
        if terminate_backwards and (np.abs(theta_bounce) > 90):
            print('ray bounced backwards, terminating integration')
            return None,None,None
        
        # update ray angle
        y_intermediate[2] = np.sin(np.radians(theta_bounce)) / c
        
        loop_count += 1

    # unlink all shared arrays after process is done
    for var in existing_shms:
        existing_shms[var].close()

    return full_ray, n_bottom, n_surface

def _init_shared_memory(
        cin,
        cpin,
        rin,
        zin,
        depths,
        depth_ranges,
        bottom_angle,
        x_eval
):
    '''
    Initialize shared memory for multiprocessing

    Parameters
    ----------
    cpin : np.array (m,n)
        2D array of dc/dz
    rin : np.array (m,)
        range coordinate for c arrays
    zin : np.array (n,)
        depth coordinate for c arrays
    depths : np.array(m,)
        array of depths (meters), should be 1D with shape (k,)
    depth_ranges : np.array(k,)
        array of depth ranges (meters), should be 1D with shape (k,)
    bottom_angle : np.array (k,)
        array of bottom angles (degrees), should be 1D with shape (k,) and values should align with depth_ranges coordinate.

    '''
    shared_array_names = [
        'cin','cpin','rin','zin','depths','depth_ranges','bottom_angle','x_eval'
    ]
    shared_arrays_np = {
        'cin':cin,
        'cpin':cpin,
        'rin':rin,
        'zin':zin,
        'depths':depths,
        'depth_ranges':depth_ranges,
        'bottom_angle':bottom_angle,
        'x_eval':x_eval
    }

    shms = {}
    shared_arrays = {}
    array_metadata = {}
    # clean up shared arrays
    _cleanup_shared_memory(shared_array_names)

    for var in shared_arrays_np:
        shms[var] = shared_memory.SharedMemory(create=True, size=shared_arrays_np[var].nbytes, name=var)
        shared_arrays[var] = np.ndarray(shared_arrays_np[var].shape, dtype=shared_arrays_np[var].dtype, buffer=shms[var].buf)
        shared_arrays[var][:] = shared_arrays_np[var][:]
        array_metadata[var] = {
            'shape': shared_arrays_np[var].shape,
            'dtype': shared_arrays_np[var].dtype,
        }
    
    return array_metadata, shms

def _cleanup_shared_memory(names):
    """
    Clean up existing shared memory objects by names
    
    Parameters
    ----------
    names : list
        names of shared memory objects to clean up
    """
    for name in names:
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
            #print(f"Cleaned up existing shared memory: {name}")
        except FileNotFoundError:
            # Memory doesn't exist, which is fine
            pass
        except Exception as e:
            print(f"Error cleaning up {name}: {e}")

def _unpack_shared_memory(shared_array_metadata):
    """
    Unpack shared memory arrays from metadata

    Parameters
    ----------
    shared_array_metadata : dict
        Dictionary containing metadata of shared memory arrays

    Returns
    -------
    shared_arrays : dict
        Dictionary containing unpacked shared memory arrays
    existing_shms : dict
        Dictionary containing existing shared memory objects
    """
    shared_array_names = [
        'cin','cpin','rin','zin','depths','depth_ranges','bottom_angle','x_eval'
    ]

    existing_shms = {}
    shared_arrays = {}
    for var in shared_array_names:
        existing_shms[var] = shared_memory.SharedMemory(name=var)

        shared_arrays[var] = np.ndarray(
            shared_array_metadata[var]['shape'],
            dtype=shared_array_metadata[var]['dtype'], buffer=existing_shms[var].buf)
        
    return shared_arrays, existing_shms

@numba.njit(fastmath=True, cache=True)
def _ray_angle(
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

__all__ = ['bilinear_interp', 'linear_interp', 'derivsrd_sp',
           'bottom_bounce_sp', 'surface_bounce_sp', 'shoot_ray_segment', 'shoot_ray', 'shoot_single_ray', '_init_shared_memory', '_unpack_shared_memory']