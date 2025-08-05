import numpy as np
import scipy.integrate
import pygenray as pr
import scipy
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import time

def shoot_rays(
        source_depth : float,
        source_range : float,
        launch_angles : np.array,
        receiver_range : float,
        num_range_save : int,
        environment : pr.OceanEnvironment2D,
        rtol = 1e-9,
        terminate_backwards : bool = True,
        n_processes : int = None,
        debug : bool = True,
        flatearth : bool = True
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
    receiver_range : float
        receiver range (meters)
    num_range_save : int
        The number of range values to save the ray state at. This value is unrelated to the numerical integration.
        The ray state value that is at the end bounds of the range integration is saved exactly.
        All other values are interpolated to a range grid with `num_range_save` points between the source and receiver range.
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
    flatearth : bool
        whether to transform environment to flat earth coordinates. Default is True.

    Returns
    -------
    ray : np.array
        2D array of ray state at each x_eval point, shape (4, n_eval), where n_eval is the number of evaluation points
    n_bott : int
        number of bottom bounces
    n_surf : int
        number of surface bounces
    '''

    # flip launch angles to match sign convention
    launch_angles = -launch_angles

    if n_processes == None:
        n_processes = mp.cpu_count()
    # set up initial conditions for ray variable

    ## unpack environment object
    cin, cpin, rin, zin, depths, depth_ranges, bottom_angles = _unpack_envi(environment, flatearth=flatearth)
    
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
                    receiver_range,
                    num_range_save,
                    environment,
                    rtol=rtol,
                    terminate_backwards=terminate_backwards,
                    debug=debug,
                    flatearth=flatearth
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
        array_metadata, shms = pr._init_shared_memory(cin, cpin, rin ,zin, depths, depth_ranges, bottom_angles)

        try:
            # calculate initial ray parameter
            c = pr.bilinear_interp(source_range, source_depth, rin, zin, cin)
            y0s = [np.array([0, source_depth, np.sin(np.radians(launch_angle))/c]) for launch_angle in launch_angles]

            shoot_ray_part = partial(
                _shoot_single_ray_process,
                source_range=source_range,
                source_depth=source_depth,
                receiver_range=receiver_range,
                num_range_save=num_range_save,
                array_metadata=array_metadata,
                rtol=rtol,
                terminate_backwards=terminate_backwards
            )
            
            with mp.Pool(n_processes) as pool:
                rays_ls = list(tqdm(pool.imap(shoot_ray_part, y0s), total=len(y0s), desc="Processing rays"))

            ranges = np.linspace(source_range, receiver_range, num_range_save)

            # unpack results
            rays_list = []
            rays_list_idx = 0  # Add separate counter for rays_list
            for k, single_ray in enumerate(rays_ls):
                if single_ray is None:
                    continue
                else:
                    # reinterpolate ray to range grid
                    rays_list.append(single_ray)

                    # _shoot_single_ray_process does not save launch angle in ray object
                    # need to set manually here
                    rays_list[rays_list_idx].launch_angle = launch_angles[k]  # Use separate counter
                    rays_list_idx += 1  # Increment counter
            
            ray_fan = pr.RayFan(rays_list)
        
        finally:
            time.sleep(0.1)  # Ensure all processes have finished before cleaning up shared memory
            # Always clean up shared memory, even if an error occurs
            for var in shms:
                try:
                    shms[var].unlink()
                    shms[var].close()
                except:
                    pass # ignore cleanup errors

        return ray_fan

def shoot_ray(
    source_depth : float,
    source_range : float,
    launch_angle : float,
    receiver_range : float,
    num_range_save : int,
    environment : pr.OceanEnvironment2D,
    rtol = 1e-9,
    terminate_backwards : bool = True,
    debug : bool = True,
    flatearth : bool = True
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
    receiver_range : float
        receiver range (meters)
    num_range_save : int
        The number of range values to save the ray state at. This value is unrelated to the numerical integration.
        The ray state value that is at the end bounds of the range integration is saved exactly.
        All other values are interpolated to a range grid with `num_range_save` points between the source and receiver range.
    environment : pr.OceanEnvironment
        OceanEnvironment object specifying sound speed and bathymetry.
    rtol : float
        relative tolerance for the ODE solver, default is 1e-6
    terminate_backwards : bool
        whether to terminate ray if it bounces backwards)
    debug : bool
        whether to print debug information, default is False
    flatearth : bool
        whether to transform environment to flat earth coordinates. Default is True.
        
    Returns 
    -------
    ray : pr.Ray
        pr.Ray object

    """

    # flip launch angle to match sign convention
    launch_angle = -launch_angle
    
    cin, cpin, rin, zin ,depths, depth_ranges, bottom_angles = _unpack_envi(environment, flatearth=flatearth)

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
        y0, source_depth, source_range, receiver_range, cin, cpin, rin, zin, depths, depth_ranges, bottom_angles, rtol, terminate_backwards,debug
    )

    if full_ray is None:
        return None
    else:
        # reinterpolate ray to range grid
        range_save = np.linspace(source_range, receiver_range, num_range_save)
        full_ray = _interpolate_ray(sols, range_save)
        ray = pr.Ray(full_ray[0,:], full_ray[1:,:], n_bottom, n_surface, launch_angle, source_depth)
    
        return ray

def _shoot_ray_array(
    y0 : np.array,
    source_depth : float,
    source_range : float,
    receiver_range : float,
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
    Integrate single ray. Integration is terminated at bottom and surface reflections, and reflection angle is calculated and updated. Integration is looped until ray reaches receiver range. If there is an error in the integration, the function returns None, None, None, None.
    
    Environment specified by numpy arrays that are returned by {mod}`pr._unpack_envi()`.

    Parameters
    ----------
    y0 : np.array (3,)
        initial ray vector values [travel time, depth, ray parameter (sin(θ)/c)].
    source_depth : float
        array of source depths (meters), should be 1D with shape (m,)
    source_range : np.array
        array of source ranges (meters), should be 1D with shape (n,)
    receiver_range : float
        receiver range (meters)
    launch_angle : float
        launch angle of ray (degrees)
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
    full_ray = np.concatenate((np.array([source_range]), y0)).copy()
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

    while x_intermediate < receiver_range:

        sol = _shoot_ray_segment(
            x_intermediate,
            y_intermediate,
            receiver_range,
            cin,
            cpin,
            rin,
            zin,
            depths,
            depth_ranges,
            rtol=rtol,
        )

        if len(sol.t) == 0:
            raise Exception('Integration segment failed, no points returned.')
        
        sols.append(sol)
        full_ray = np.append(full_ray, np.vstack((sol.t, sol.y)), axis=1)

        # if end of integration is reached, end loop
        if sol.status == 0:
            break
        elif sol.status == -1:
            if debug:
                print(f'Integration failed with message: {sol.message}')
            return None, None, None, None

        y_intermediate = sol.y[:,-1]

        # Bounce Event
        if len(sol.t_events[0]) > 0 or len(sol.t_events[1]) > 0:
            # Bounce event occurred, use the event time as the new range
            if len(sol.t_events[0]) > 0:  # Surface event
                x_intermediate = sol.t_events[0][0]
            elif len(sol.t_events[1]) > 0:  # Bottom event  
                x_intermediate = sol.t_events[1][0]
        
        # Vertical Ray
        elif len(sol.t_events[2]) > 0:  # Vertical ray event
            if debug:
                print(f'ray is vertical at x={sol.t[-1]}, y={sol.y[1,-1]}, terminating integration')
            return None, None, None, None

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
        receiver_range : float,
        num_range_save : int,
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
    receiver_range : float
        integration range end bound. starting point is x0
    num_range_save : int
        The number of range values to save the ray state at. This value is unrelated to the numerical integration.
        The ray state value that is at the end bounds of the range integration is saved exactly.
        All other values are interpolated to a range grid with `num_range_save` points between the source and receiver range.
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

    try:
        # Access shared arrays
        shared_arrays, existing_shms = pr._unpack_shared_memory(array_metadata)

        cin = shared_arrays['cin']
        cpin = shared_arrays['cpin']
        rin = shared_arrays['rin']
        zin = shared_arrays['zin']
        depths = shared_arrays['depths']
        depth_ranges = shared_arrays['depth_ranges']
        bottom_angles = shared_arrays['bottom_angle']

        sols, full_ray, n_bottom, n_surface = _shoot_ray_array(
            y0,
            source_depth,
            source_range,
            receiver_range,
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
        
        range_save = np.linspace(source_range, receiver_range, num_range_save)

        if full_ray is None:
            return None
        else:
            # reinterpolate ray to range grid

            full_ray_interpolated = _interpolate_ray(sols, range_save)  

            ray = pr.Ray(
                full_ray_interpolated[0,:],
                full_ray_interpolated[1:,:],
                n_bottom,
                n_surface,
                source_depth=source_depth
            )
    except Exception as e:
        if debug:
            print(f'Error in ray integration: {e}')
        return None
    finally:
        # Always close shared memory handles, even with error
        for var in existing_shms:
            try:
                existing_shms[var].close()
            except:
                pass # ignore cleanup errors

    return ray

def _shoot_ray_segment(
        x0 : float,
        y0 : np.array,
        receiver_range : float,
        cin : np.array,
        cpin : np.array,
        rin : np.array,
        zin : np.array,
        depths : np.array,
        depth_ranges : np.array,
        rtol = 1e-9,
        **kwargs
):
    """
    Given an initial condition vector and initial range, integrate ray
    until integration bounds or event is triggered (such as surface or bottom bounce).

    any keyword arguments are passed to {mod}`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    x0 : float
        initial ray, x position
    y : np.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    receiver_range : float
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
    **kwargs : dict
        kwargs passed to {mod}`scipy.integrate.solve_ivp`.

    Returns
    -------
    sol : scipy.integrate.OdeResult
        solution to integration segment.
    """

    # set up surface and bottom bounce events
    surface_event = pr.surface_bounce
    surface_event.terminal = True
    surface_event.direction = 1
    
    bottom_event = pr.bottom_bounce  
    bottom_event.terminal = True
    bottom_event.direction = 1

    vertical_ray = pr.vertical_ray
    vertical_ray.terminal = True

    events = (
        surface_event,
        bottom_event,
        vertical_ray,
    )

    sol = scipy.integrate.solve_ivp(
        pr.derivsrd,
        (x0,receiver_range),
        y0,
        args = (cin, cpin, rin, zin, depths, depth_ranges),
        events = events,
        rtol = rtol,
        dense_output=True,
        **kwargs
    )

    return sol

def _unpack_envi(environment, flatearth=True):

    if flatearth:
        # chech that environment.sound_speed_fe exists
        if not hasattr(environment, 'sound_speed_fe'):
            raise Exception('Flat earth transformation has not been applied. Set `flat_earth_transform=True` when creating the OceanEnvironment2D object.')
        
        cin = np.array(environment.sound_speed_fe.values)
        cpin = np.array(environment.sound_speed_fe.differentiate('depth').values)
        rin = np.array(environment.sound_speed_fe.range.values)
        zin = np.array(environment.sound_speed_fe.depth.values)
        depths = np.array(environment.bathymetry_fe.values)
        depth_ranges = np.array(environment.bathymetry_fe.range.values)
        bottom_angles = np.array(environment.bottom_angle)
    else:
        cin = np.array(environment.sound_speed.values)
        cpin = np.array(environment.sound_speed.differentiate('depth').values)
        rin = np.array(environment.sound_speed.range.values)
        zin = np.array(environment.sound_speed.depth.values)
        depths = np.array(environment.bathymetry.values)
        depth_ranges = np.array(environment.bathymetry.range.values)
        bottom_angles = np.array(environment.bottom_angle)

    return cin, cpin, rin, zin ,depths, depth_ranges, bottom_angles


def _interpolate_ray(
        sols : list,
        range_save : np.array
):
    """
    Given list of {mod}`scipy.integrate._ivp.ivp.OdeResult` solutions, corresponding to integrated ray segments
    interpolate ray state to specfied range grid using the order of the numerical solver (using `dense_output=True` in `scipy.integrate.solve_ivp`).

    Parameters
    ----------
    sols : list
        List of {mod}`scipy.integrate._ivp.ivp.OdeResult` solutions, corresponding to integrated ray segments
    range_save : np.array (m,)
        array of range values to save the ray state at

    Returns
    -------
    full_ray_state : np.array (4,m)
        4D array of ray state at each range_save point first dimension corresponds to [range, time, depth, pz], m is the number of range values to save
    """

    full_ray = np.ones((3, len(range_save)-1))*np.nan

    for k, sol in enumerate(sols):
        idx1 = np.argmin(np.abs(range_save - sol.t[0]))
        idx2 = np.argmin(np.abs(range_save - sol.t[-1]))

        full_ray[:, idx1:idx2] = sol.sol(range_save[idx1:idx2])

    # Append final ray state to full_ray
    full_ray = np.concatenate((full_ray, np.expand_dims(sols[-1].y[:,-1], axis=1)), axis=1)

    # Append range values to full_ray
    full_ray_state = np.concatenate((np.expand_dims(range_save, axis=0), full_ray), axis=0)

    return full_ray_state


__all__ = ['_shoot_ray_segment', 'shoot_rays', 'shoot_ray','_shoot_single_ray_process', '_unpack_envi', '_shoot_ray_array']
