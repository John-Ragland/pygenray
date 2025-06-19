import numpy as np
import scipy.integrate
import pygenray as pr
import scipy
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def shoot_rays(
        source_depth : float,
        source_range : float,
        launch_angles : np.array,
        reciever_range : float,
        x_eval : np.array,
        environment : pr.OceanEnvironment2D,
        rtol = 1e-9,
        terminate_backwards : bool = True,
        n_processes : int = None,
        debug : bool = True
):
    '''
    Integrate rays for given environment and launch angles. Different launch angle initial conditions are mapped to available CPUS.

    Parameters
    ----------
    source_depth : np.array
        array of source depths (meters)
    source_range : np.array
        array of source ranges (meters)
    launch_angles : np.array
        array of source angles (degrees)
    reciever_range : float
        reciever range (meters)
    x_eval : np.array
        The range values to save the ray state at.s
    environment : pr.OceanEnvironment
        OceanEnvironment object specifying sound speed and bathymetry.
    rtol : float
        relative tolerance for the ODE solver, default is 1e-6
    terminate_backwards : bool
        whether to terminate ray if it bounces backwards
    n_processes : int
        number of processes to use, Default of None (mp.cpu_count)
    debug : bool
        whether to print debug information, default is False

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
    cin, cpin, rin, zin, depths, depth_ranges, bottom_angles = _unpack_envi(environment)
    
    # check that coordinates are monotonically increasing
    if not (np.all(np.diff(rin) >= 0)):
        raise Exception('Sound speed range coordinates must be monotonically increasing.')
    if not (np.all(np.diff(zin) >= 0)):
        raise Exception('Sound speed depth coordinates must be monotonically increasing.')
    if not (np.all(np.diff(depth_ranges) >= 0)):
        raise Exception('Bathymetry range coordinates must be monotonically increasing.')

    # Use multiprocessing if number of rays is high enough
    # TODO set threshold to accurately reflect overhead trade off
    if len(launch_angles) < 70:
        rays_ls = []
        for launch_angle in tqdm(launch_angles):
            rays_ls.append(
                shoot_ray(
                    source_depth,
                    source_range,
                    launch_angle,
                    reciever_range,
                    x_eval,
                    environment,
                    rtol=rtol,
                    terminate_backwards=terminate_backwards,
                    debug=debug
                )
            )
        # shoot_ray automatically saves launch angle to ray object
        # launch angle doesn't need to be set manually here

        # remove dropped rays
        rays_ls_nonone = [ray for ray in rays_ls if ray is not None]
        
        rays = pr.RayFan(rays_ls_nonone)
        return rays
    else: # Use multiprocessing
        # Create Shared Arrays
        array_metadata, shms = pr._init_shared_memory(cin, cpin, rin ,zin, depths, depth_ranges, bottom_angles, x_eval)

        # calculate initial ray parameter
        c = pr.bilinear_interp(source_range, source_depth, rin, zin, cin)
        y0s = [np.array([0, source_depth, np.sin(np.radians(launch_angle))/c]) for launch_angle in launch_angles]

        shoot_ray_part = partial(
            _shoot_single_ray_process,
            source_range=source_range,
            source_depth=source_depth,
            reciever_range=reciever_range,
            array_metadata=array_metadata,
            rtol=rtol,
            terminate_backwards=terminate_backwards
        )
        
        with mp.Pool(n_processes) as pool:
            rays_ls = list(tqdm(pool.imap(shoot_ray_part, y0s), total=len(y0s), desc="Processing rays"))

        # unpack results
        rays_list = []
        for k, single_ray in enumerate(rays_ls):
            if single_ray is None:
                continue
            else:
                rays_list.append(single_ray)

                # _shoot_single_ray_process does not save launch angle in ray object
                # need to set manually here
                rays_list[k].launch_angle = launch_angles[k]

        ray_fan = pr.RayFan(rays_list)

        # close and unlink shared memory
        for var in shms:
            shms[var].unlink()
            shms[var].close()

        return ray_fan

def shoot_ray(
    source_depth : float,
    source_range : float,
    launch_angle : float,
    reciever_range : float,
    x_eval : np.array,
    environment : pr.OceanEnvironment2D,
    rtol = 1e-9,
    terminate_backwards : bool = True,
    debug : bool = True
):
    """
    Integrate rays for given environment and launch angles. Different launch angle initial conditions are mapped to available CPUS.
    
    Parameters
    ----------
    source_depth : float
        array of source depths (meters)
    source_range : float
        array of source ranges (meters)
    launch_angle : np.array
        array of source angles (degrees), should be 1D with shape (k,)
    reciever_range : float
        reciever range (meters)
    x_eval : np.array
        The range values to save the ray state at.s
    environment : pr.OceanEnvironment
        OceanEnvironment object specifying sound speed and bathymetry.
    rtol : float
        relative tolerance for the ODE solver, default is 1e-6
    terminate_backwards : bool
        whether to terminate ray if it bounces backwards)
    debug : bool
        whether to print debug information, default is False

    """
    cin, cpin, rin, zin ,depths, depth_ranges, bottom_angles = _unpack_envi(environment)

    # check that coordinates are monotonically increasing
    if not (np.all(np.diff(rin) >= 0)):
        raise Exception('Sound speed range coordinates must be monotonically increasing.')
    if not (np.all(np.diff(zin) >= 0)):
        raise Exception('Sound speed depth coordinates must be monotonically increasing.')
    if not (np.all(np.diff(depth_ranges) >= 0)):
        raise Exception('Bathymetry range coordinates must be monotonically increasing.')
    
    # calculate y0
    c = pr.bilinear_interp(source_range, source_depth, rin, zin, cin)
    y0 = np.array([0, source_depth, np.sin(np.radians(launch_angle))/c])

    # launch ray at angle theta
    sols, full_ray, n_bottom, n_surface = _shoot_ray_array(
        y0, source_depth, source_range, reciever_range, x_eval, cin, cpin, rin, zin, depths, depth_ranges, bottom_angles, rtol, terminate_backwards,debug
    )
    if full_ray is None:
        return None
    else:
        ray = pr.Ray(full_ray[0,:], full_ray[1:,:], n_bottom, n_surface, launch_angle, source_depth)

        return ray

def _shoot_ray_array(
    y0 : np.array,
    source_depth : float,
    source_range : float,
    reciever_range : float,
    x_eval : np.array,
    cin : np.array,
    cpin : np.array,
    rin : np.array,
    zin : np.array,
    depths : np.array,
    depth_ranges : np.array,
    bottom_angles : np.array,
    rtol = 1e-9,
    terminate_backwards : bool = True,
    debug : bool = True,
):
    """
    Integrate single ray. Integration is terminated at bottom and surface reflections, and reflection angle is calculated and updated. Integration is looped until ray reaches reciever range. If there is an error in the integration, the function returns None, None, None.
    
    Environment specified by numpy arrays that are returned by {mod}`pr._unpack_envi()`.

    Parameters
    ----------
    y0 : np.array (3,)
        initial ray vector values [travel time, depth, ray parameter (sin(θ)/c)].
    source_depth : float
        array of source depths (meters), should be 1D with shape (m,)
    source_range : np.array
        array of source ranges (meters), should be 1D with shape (n,)
    reciever_range : float
        reciever range (meters)
    launch_angle : float
        launch angle of ray (degrees)
    x_eval : np.array
        The range values to save the ray state at.s
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
    bottom_angles : np.array(m,)
        array of bottom angles (degrees), should be 1D with shape and correspond to range bins `depth_ranges`
    environment : pr.OceanEnvironment
        OceanEnvironment object specifying sound speed and bathymetry.
    rtol : float
        relative tolerance for the ODE solver, default is 1e-6
    terminate_backwards : bool
        whether to terminate ray if it bounces backwards. Default True.
    debug : bool
        whether to print debug information, default is False

    Returns
    -------
    ray : pr.Ray
        Ray object
    """

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
        bottom_angles,
        kind='cubic'
    )

    # set intermediate ray state to initial ray state
    y_intermediate = y0.copy()

    while x_intermediate < reciever_range:

        x_eval_filtered = x_eval[x_eval >= x_intermediate]

        sol = _shoot_ray_segment(
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
        )

        if len(sol.t) == 0:
            raise Exception('Integration segment failed, no points returned.')
        
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
        theta,c = pr.ray_angle(x_intermediate, y_intermediate, cin, rin, zin)
        
        # Surface Bounce
        if len(sol.t_events[0])==1:
            theta_bounce = -theta
            n_surface += 1

        # Bottom Bounce
        elif len(sol.t_events[1])==1:
            # β: bottom angle in degrees
            beta = bottom_angle_interp(x_intermediate)
            theta_bounce = 2*beta - theta
            n_bottom += 1

        # terminate if ray bounces backwards
        if terminate_backwards and (np.abs(theta_bounce) > 90):
            if debug:
                print(f'ray bounced backwards, terminating integration')
            return None,None,None,None
        
        # update ray angle
        y_intermediate[2] = np.sin(np.radians(theta_bounce)) / c
        
        loop_count += 1
        
    return sols, full_ray, n_bottom, n_surface

def _shoot_single_ray_process(
        y0 : np.array,
        source_range : float,
        source_depth : float,
        reciever_range : float,
        array_metadata : dict,
        rtol = 1e-9,
        terminate_backwards : bool = True,
        debug : bool = False
):
    """
    Shoot a single ray, accessing shared memory for environment data.
    This is an internal function for multiprocessing handling.

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
    debug : bool
        whether to print debug information, default is False

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
    shared_arrays, existing_shms = pr._unpack_shared_memory(array_metadata)

    cin = shared_arrays['cin']
    cpin = shared_arrays['cpin']
    rin = shared_arrays['rin']
    zin = shared_arrays['zin']
    depths = shared_arrays['depths']
    depth_ranges = shared_arrays['depth_ranges']
    bottom_angles = shared_arrays['bottom_angle']
    x_eval = shared_arrays['x_eval']

    sols, full_ray, n_bottom, n_surface = _shoot_ray_array(
        y0,
        source_depth,
        source_range,
        reciever_range,
        x_eval,
        cin,
        cpin,
        rin,
        zin,
        depths,
        depth_ranges,
        bottom_angles,
        rtol,
        terminate_backwards,
        debug,
    )
    
    if full_ray is None:
        return None
    else:
        ray = pr.Ray(full_ray[0,:], full_ray[1:,:], n_bottom, n_surface, source_depth=source_depth)

    # unlink all shared arrays after process is done
    for var in existing_shms:
        existing_shms[var].close()

    return ray

def _shoot_ray_segment(
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
        rtol = 1e-9,
        debug : bool = False
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
    debug : bool
        whether to print debug information, default is False
    """

    # set up surface and bottom bounce events
    surface_event = pr.surface_bounce
    surface_event.terminal = True
    surface_event.direction = 1
    
    bottom_event = pr.bottom_bounce  
    bottom_event.terminal = True
    bottom_event.direction = 1

    events = (
        surface_event,
        bottom_event,
    )
    
    sol = scipy.integrate.solve_ivp(
        pr.derivsrd,
        (x0,reciever_range),
        y0,
        args = (cin, cpin, rin*1000, zin, depths, depth_ranges),
        t_eval=x_eval,
        events = events,
        rtol = rtol
    )

    return sol

def _unpack_envi(environment):
    cin = np.array(environment.sound_speed.values)
    cpin = np.array(environment.sound_speed.differentiate('depth').values)
    rin = np.array(environment.sound_speed.range.values)
    zin = np.array(environment.sound_speed.depth.values)
    depths = np.array(environment.bathymetry.values)
    depth_ranges = np.array(environment.bathymetry.range.values)
    bottom_angles = np.array(environment.bottom_angle)

    return cin, cpin, rin, zin ,depths, depth_ranges, bottom_angles

__all__ = ['_shoot_ray_segment', 'shoot_rays', 'shoot_ray','_shoot_single_ray_process', '_unpack_envi', '_shoot_ray_array']