import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import diffrax
import optimistix as optx
import pygenray as pr
from tqdm import tqdm

from .integration_processes import derivsrd, bilinear_interp, linear_interp, ray_angle, event_cond


def shoot_rays(
        source_depth: float,
        source_range: float,
        launch_angles: np.array,
        receiver_range: float,
        num_range_save: int,
        environment: pr.OceanEnvironment2D,
        rtol=1e-9,
        terminate_backwards: bool = True,
        n_processes: int = None,  # deprecated, ignored
        debug: bool = True,
        flatearth: bool = True,
        parallel: bool = True,
        max_bounces: int = 50,
):
    '''
    Integrate rays for given environment and launch angles.

    Parameters
    ----------
    source_depth : float
        source depth (meters)
    source_range : float
        source range (meters)
    launch_angles : np.array
        array of source angles (degrees)
    receiver_range : float
        receiver range (meters)
    num_range_save : int
        number of range points to save ray state at
    environment : pr.OceanEnvironment2D
        OceanEnvironment object specifying sound speed and bathymetry
    rtol : float
        relative tolerance for the ODE solver, default is 1e-9
    terminate_backwards : bool
        whether to terminate ray if it bounces backwards
    n_processes : int
        deprecated, ignored
    debug : bool
        whether to print debug information (serial path only)
    flatearth : bool
        whether to transform environment to flat earth coordinates
    parallel : bool
        if True (default), use jax.vmap for parallel ray tracing;
        if False, use serial tqdm loop
    max_bounces : int
        maximum number of boundary bounces per ray (vmap path only)

    Returns
    -------
    rays : pr.RayFan
    '''
    if type(launch_angles) is list:
        launch_angles = np.array(launch_angles)

    cin, cpin, rin, zin, depths, depth_ranges, bottom_angles = _unpack_envi(environment, flatearth=flatearth)

    if not (np.all(np.diff(rin) >= 0)):
        raise Exception('Sound speed range coordinates must be monotonically increasing.')
    if not (np.all(np.diff(zin) >= 0)):
        raise Exception('Sound speed depth coordinates must be monotonically increasing.')
    if not (np.all(np.diff(depth_ranges) >= 0)):
        raise Exception('Bathymetry range coordinates must be monotonically increasing.')

    cin_j = jnp.array(cin)
    cpin_j = jnp.array(cpin)
    rin_j = jnp.array(rin)
    zin_j = jnp.array(zin)
    depths_j = jnp.array(depths)
    dr_j = jnp.array(depth_ranges)
    bottom_angles_j = jnp.array(bottom_angles)
    env_arrays = (cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j, bottom_angles_j)

    range_save = np.linspace(source_range, receiver_range, num_range_save)

    if parallel:
        rays = _shoot_rays_vmap(
            launch_angles,
            source_depth, source_range, receiver_range,
            range_save, env_arrays,
            max_bounces=max_bounces, rtol=rtol,
            terminate_backwards=terminate_backwards,
        )
        return rays

    # Serial fallback
    launch_angles_flipped = -launch_angles  # flip to match sign convention

    rays_ls = []
    for launch_angle in tqdm(launch_angles_flipped, desc="Computing ray fan"):
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
                flatearth=flatearth,
                _env_arrays=env_arrays,
            )
        )

    rays_ls_nonone = [ray for ray in rays_ls if ray is not None]
    rays = pr.RayFan(rays_ls_nonone)
    return rays


def _shoot_rays_vmap(
    launch_angles,       # (K,) user-convention degrees (NOT sign-flipped)
    source_depth,
    source_range,
    receiver_range,
    range_save,          # (N,) numpy array
    env_arrays,
    max_bounces=50,
    rtol=1e-9,
    terminate_backwards=True,
):
    """
    vmap-parallel ray fan computation.

    Parameters
    ----------
    launch_angles : array (K,)
        Launch angles in degrees, user convention (positive = toward surface).
        These are NOT pre-flipped — the sign convention is handled internally.
    source_depth, source_range, receiver_range : float
    range_save : (N,) numpy array of range points
    env_arrays : tuple of JAX arrays
    max_bounces, rtol, terminate_backwards : passed to _shoot_single_ray_jax

    Returns
    -------
    pr.RayFan of valid rays
    """
    # Convert to JAX arrays
    launch_angles_rad = jnp.radians(jnp.array(launch_angles, dtype=jnp.float64))
    range_save_j = jnp.array(range_save, dtype=jnp.float64)
    source_depth_j = jnp.float64(source_depth)
    source_range_j = jnp.float64(source_range)
    receiver_range_j = jnp.float64(receiver_range)

    # Build a partial with static args baked in, then vmap over launch_angles_rad
    _single = functools.partial(
        _shoot_single_ray_jax,
        source_depth=source_depth_j,
        source_range=source_range_j,
        receiver_range=receiver_range_j,
        range_save=range_save_j,
        env_arrays=env_arrays,
        max_bounces=max_bounces,
        rtol=rtol,
        terminate_backwards=terminate_backwards,
    )
    vmapped = jax.vmap(_single, in_axes=(0,))
    outputs, n_bottoms, n_surfaces, alives, reacheds = vmapped(launch_angles_rad)

    # Convert to numpy for Ray construction
    outputs_np = np.array(outputs)
    n_bottoms_np = np.array(n_bottoms)
    n_surfaces_np = np.array(n_surfaces)
    reacheds_np = np.array(reacheds)
    launch_angles_np = np.asarray(launch_angles)

    rays_ls = []
    for k in range(len(launch_angles_np)):
        if not reacheds_np[k]:
            continue
        y = outputs_np[k].T  # (3, N) in ODE format: [travel_time, z_ode, p_ode]
        ray = pr.Ray(
            r=range_save,
            y=y,
            n_bottom=int(n_bottoms_np[k]),
            n_surface=int(n_surfaces_np[k]),
            launch_angle=float(launch_angles_np[k]),
            source_depth=float(source_depth),
        )
        rays_ls.append(ray)

    return pr.RayFan(rays_ls)


@functools.partial(jax.jit, static_argnames=['rtol', 'max_bounces', 'terminate_backwards'])
def _shoot_single_ray_jax(
    launch_angle_rad,       # radians of user-convention angle; y0[2] = sin(launch_angle_rad)/c
    source_depth,
    source_range,
    receiver_range,
    range_save,             # (N,) float64 JAX array
    env_arrays,             # (cin, cpin, rin, zin, depths, dr, bottom_angles)
    max_bounces=50,         # static
    rtol=1e-9,              # static
    terminate_backwards=True,  # static
):
    """
    Integrate a single ray using lax.while_loop; vmappable.

    Parameters
    ----------
    launch_angle_rad : scalar float64
        Launch angle in radians using user convention. Internally:
        y0[2] = sin(launch_angle_rad) / c0. Matches the effective angle
        used by shoot_rays (which applies a double sign-flip).
    source_depth, source_range, receiver_range : scalar float64
    range_save : (N,) float64 JAX array
    env_arrays : tuple (cin, cpin, rin, zin, depths, dr, bottom_angles) as JAX arrays
    max_bounces : int (static)
    rtol : float (static)
    terminate_backwards : bool (static)

    Returns
    -------
    output : (N, 3) float64 — [travel_time, z_ode, p_ode] at each range_save point
    n_bottom : int32
    n_surface : int32
    alive : bool — False if ray went vertical or backwards
    reached : bool — True if ray reached receiver range
    """
    cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j, bottom_angles_j = env_arrays
    args = (cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j)

    c0 = bilinear_interp(source_range, source_depth, rin_j, zin_j, cin_j)
    p0 = jnp.sin(launch_angle_rad) / c0
    y0 = jnp.array([jnp.float64(0.0), source_depth, p0])

    N = range_save.shape[0]
    output = jnp.full((N, 3), jnp.nan, dtype=jnp.float64)

    init_carry = (
        jnp.float64(source_range),   # x_current
        y0,                           # y_current (3,)
        output,                       # output (N, 3)
        jnp.int32(0),                 # n_bottom
        jnp.int32(0),                 # n_surface
        jnp.bool_(True),              # alive
        jnp.int32(0),                 # bounce_count
    )

    def _cond_fn(carry):
        x_current, y_current, out, n_bottom, n_surface, alive, bounce_count = carry
        return alive & (x_current < receiver_range - 1.0) & (bounce_count < max_bounces)

    def _body_fn(carry):
        x_current, y_current, out, n_bottom, n_surface, alive, bounce_count = carry

        # Clamp range_save to [x_current, receiver_range] for fixed-shape ts
        ts_clamped = jnp.clip(range_save, x_current, receiver_range)

        # Run ODE segment with SubSaveAt for fixed-shape output
        sol = _shoot_ray_segment_fixed(
            x_current, y_current, receiver_range, args, ts_clamped, rtol
        )

        # actual_x1: where the event fired (or receiver_range if no event)
        actual_x1 = sol.ts[0][1]
        y_end = sol.ys[0][1]  # state at actual_x1, shape (3,)

        # Accumulate output: only overwrite valid range points
        is_valid = (range_save >= x_current) & (range_save <= actual_x1)
        out = jnp.where(is_valid[:, None], sol.ys[1], out)

        # Classify termination
        c_end = bilinear_interp(actual_x1, y_end[1], rin_j, zin_j, cin_j)
        theta_end = jnp.degrees(jnp.arcsin(jnp.clip(y_end[2] * c_end, -1.0, 1.0)))
        bd_end = linear_interp(actual_x1, dr_j, depths_j)

        at_receiver = actual_x1 >= receiver_range - 1.0
        is_vertical = jnp.abs(theta_end) >= (90.0 - 1e-3) - 0.01
        is_surface = y_end[1] <= 1.0
        is_at_bottom = y_end[1] >= bd_end - 1.0
        # Unexpected termination: not at receiver, not any recognized event
        is_unexpected = ~at_receiver & ~is_surface & ~is_vertical & ~is_at_bottom

        # Compute bounce angles (computed for all cases; only used when bouncing)
        beta = linear_interp(actual_x1, dr_j, bottom_angles_j)
        theta_surface = -theta_end
        theta_bottom = 2.0 * beta - theta_end
        theta_bounce = jnp.where(is_surface, theta_surface, theta_bottom)

        p_new = jnp.sin(jnp.radians(theta_bounce)) / c_end

        if terminate_backwards:
            is_backwards = jnp.abs(theta_bounce) > 90.0
        else:
            is_backwards = jnp.bool_(False)

        # Snap depth to boundary
        y_new_depth = jnp.where(is_surface, jnp.float64(0.0), bd_end)
        y_new = jnp.array([y_end[0], y_new_depth, p_new])

        # A valid bounce: not at receiver, not failed
        is_bounce = ~at_receiver & ~is_vertical & ~is_backwards & ~is_unexpected

        n_surface_new = n_surface + jnp.where(
            is_bounce & is_surface, jnp.int32(1), jnp.int32(0)
        )
        n_bottom_new = n_bottom + jnp.where(
            is_bounce & ~is_surface, jnp.int32(1), jnp.int32(0)
        )

        # Mark ray as dead on failure
        alive_new = alive & ~is_vertical & ~is_backwards & ~is_unexpected

        # Advance state only on valid bounce
        y_next = jnp.where(is_bounce, y_new, y_current)

        return (
            actual_x1,        # always advance x so condition can check at_receiver
            y_next,
            out,
            n_bottom_new,
            n_surface_new,
            alive_new,
            bounce_count + jnp.int32(1),
        )

    final_carry = lax.while_loop(_cond_fn, _body_fn, init_carry)
    x_final, _, output_final, n_bottom, n_surface, alive, _ = final_carry

    reached = x_final >= receiver_range - 1.0
    return output_final, n_bottom, n_surface, alive, reached


def _shoot_ray_segment_fixed(x0, y0, receiver_range, args, ts_clamped, rtol=1e-9):
    """
    Integrate a single ray segment using SubSaveAt for fixed-shape output.

    Unlike _shoot_ray_segment (which uses dense=True), this returns fixed-shape
    arrays compatible with lax.while_loop carry.

    Parameters
    ----------
    x0 : float64 scalar
    y0 : (3,) float64 array
    receiver_range : float64 scalar
    args : tuple (cin, cpin, rin, zin, depths, dr)
    ts_clamped : (N,) float64 — range_save clipped to [x0, receiver_range]
    rtol : float (compile-time constant when called inside jit)

    Returns
    -------
    sol : diffrax Solution with:
        sol.ts[0] = (2,)    [x0, x_end]
        sol.ys[0] = (2, 3)  states at x0 and x_end
        sol.ts[1] = (N,)    ts_clamped (unchanged)
        sol.ys[1] = (N, 3)  states at ts_clamped (inf past event)
    """
    event = diffrax.Event(
        event_cond,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
        direction=True,  # upcrossing only
    )

    term = diffrax.ODETerm(derivsrd)
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=rtol * 1e-3)
    saveat = diffrax.SaveAt(subs=[
        diffrax.SubSaveAt(t0=True, t1=True),  # sol.ys[0]: (2,3), sol.ts[0]: (2,)
        diffrax.SubSaveAt(ts=ts_clamped),     # sol.ys[1]: (N,3), sol.ts[1]: (N,)
    ])

    dt0 = (receiver_range - x0) * 1e-3

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=x0,
        t1=receiver_range,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=100000,
        throw=False,
    )

    return sol


def shoot_ray(
    source_depth: float,
    source_range: float,
    launch_angle: float,
    receiver_range: float,
    num_range_save: int,
    environment: pr.OceanEnvironment2D,
    rtol=1e-9,
    terminate_backwards: bool = True,
    debug: bool = True,
    flatearth: bool = True,
    _env_arrays=None,
):
    """
    Integrate a single ray for a given environment and launch angle.

    Parameters
    ----------
    source_depth : float
        source depth (meters)
    source_range : float
        source range (meters)
    launch_angle : float
        launch angle (degrees)
    receiver_range : float
        receiver range (meters)
    num_range_save : int
        number of range points to save ray state at
    environment : pr.OceanEnvironment2D
        OceanEnvironment object specifying sound speed and bathymetry
    rtol : float
        relative tolerance for the ODE solver
    terminate_backwards : bool
        whether to terminate ray if it bounces backwards
    debug : bool
        whether to print debug information
    flatearth : bool
        whether to transform environment to flat earth coordinates
    _env_arrays : tuple or None
        pre-converted JAX arrays (cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j, bottom_angles_j);
        when provided, skips _unpack_envi and validation (used by shoot_rays for performance)

    Returns
    -------
    ray : pr.Ray or None
    """
    launch_angle = -launch_angle  # flip to match sign convention

    if _env_arrays is not None:
        cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j, bottom_angles_j = _env_arrays
    else:
        cin, cpin, rin, zin, depths, depth_ranges, bottom_angles = _unpack_envi(environment, flatearth=flatearth)

        if not (np.all(np.diff(rin) >= 0)):
            raise Exception('Sound speed range coordinates must be monotonically increasing.')
        if not (np.all(np.diff(zin) >= 0)):
            raise Exception('Sound speed depth coordinates must be monotonically increasing.')
        if not (np.all(np.diff(depth_ranges)) >= 0):
            raise Exception('Bathymetry range coordinates must be monotonically increasing.')

        cin_j = jnp.array(cin)
        cpin_j = jnp.array(cpin)
        rin_j = jnp.array(rin)
        zin_j = jnp.array(zin)
        depths_j = jnp.array(depths)
        dr_j = jnp.array(depth_ranges)
        bottom_angles_j = jnp.array(bottom_angles)

    args = (cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j)

    c = pr.bilinear_interp(source_range, source_depth, rin_j, zin_j, cin_j)
    y0 = np.array([0.0, source_depth, np.sin(np.radians(launch_angle)) / float(c)])

    sols, success, n_bottom, n_surface = _shoot_ray_array(
        y0, source_range, receiver_range, args, bottom_angles_j,
        rtol, terminate_backwards, debug
    )

    if sols is None:
        return None

    range_save = np.linspace(source_range, receiver_range, num_range_save)
    full_ray = _interpolate_ray(sols, range_save)
    ray = pr.Ray(full_ray[0, :], full_ray[1:, :], n_bottom, n_surface, launch_angle, source_depth)
    return ray


def _shoot_ray_array(
    y0: np.array,
    source_range: float,
    receiver_range: float,
    args: tuple,
    bottom_angles_j,
    rtol=1e-9,
    terminate_backwards: bool = True,
    debug: bool = True,
):
    """
    Integrate single ray with bounce handling using diffrax.

    Returns
    -------
    sols : list of diffrax solutions, or None on failure
    success : True on success, None on failure
    n_bottom : int
    n_surface : int
    """
    cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j = args

    x_intermediate = source_range
    y_intermediate = y0.copy()
    sols = []
    n_surface = 0
    n_bottom = 0

    while x_intermediate < receiver_range:
        sol = _shoot_ray_segment(jnp.float64(x_intermediate), jnp.asarray(y_intermediate), jnp.float64(receiver_range), args, rtol=rtol)
        sols.append(sol)

        x_f = float(sol.ts[-1])
        y_f = np.array(sol.ys[-1])
        c_f = float(bilinear_interp(x_f, y_f[1], rin_j, zin_j, cin_j))
        theta_f = float(jnp.degrees(jnp.arcsin(jnp.clip(y_f[2] * c_f, -1., 1.))))
        bd_f = float(linear_interp(x_f, dr_j, depths_j))

        y_intermediate = y_f.copy()

        # Normal completion: reached receiver range
        if abs(x_f - receiver_range) < 1.0:
            break

        # Determine which event fired and handle bounce
        if abs(theta_f) >= (90.0 - 1e-3) - 0.01:
            if debug:
                print(f'ray is vertical at x={x_f}, terminating integration')
            return None, None, None, None
        elif y_f[1] <= 1.0:
            theta_bounce = -theta_f
            n_surface += 1
            y_intermediate[1] = 0.0  # snap exactly to surface
        elif y_f[1] >= bd_f - 1.0:
            beta = float(linear_interp(jnp.float64(x_f), dr_j, bottom_angles_j))
            theta_bounce = 2 * beta - theta_f
            n_bottom += 1
            y_intermediate[1] = bd_f  # snap exactly to bottom
        else:
            if debug:
                print(f'Integration ended unexpectedly at x={x_f}, y={y_f[1]}, terminating')
            return None, None, None, None

        if terminate_backwards and abs(theta_bounce) > 90:
            if debug:
                print(f'ray bounced backwards, terminating integration')
            return None, None, None, None

        y_intermediate[2] = np.sin(np.radians(theta_bounce)) / c_f
        x_intermediate = x_f

    return sols, True, n_bottom, n_surface


@jax.jit(static_argnames=['rtol'])
def _shoot_ray_segment(x0: float, y0, receiver_range: float, args: tuple, rtol=1e-9):
    """
    Integrate a single ray segment from x0 to receiver_range using diffrax.
    Terminates early at a surface bounce, bottom bounce, or vertical ray event.

    Parameters
    ----------
    x0 : float
        initial range
    y0 : array (3,)
        initial ray state [travel time, depth, ray parameter]
    receiver_range : float
        end range for integration
    args : tuple
        (cin, cpin, rin, zin, depths, depth_ranges) as JAX arrays
    rtol : float
        relative tolerance

    Returns
    -------
    sol : diffrax solution object with dense output
    """
    event = diffrax.Event(
        event_cond,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
        direction=True,  # upcrossing only (negative → positive)
    )

    term = diffrax.ODETerm(derivsrd)
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=rtol * 1e-3)
    saveat = diffrax.SaveAt(t0=True, t1=True, dense=True)

    y0_jax = jnp.asarray(y0)
    dt0 = (receiver_range - x0) * 1e-3

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=x0,
        t1=receiver_range,
        dt0=dt0,
        y0=y0_jax,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=100000,
        throw=False,
    )

    return sol


def _unpack_envi(environment, flatearth=True):
    if flatearth:
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

    return cin, cpin, rin, zin, depths, depth_ranges, bottom_angles


def _interpolate_ray(sols: list, range_save: np.array):
    """
    Interpolate ray state to a uniform range grid using diffrax dense output.

    Parameters
    ----------
    sols : list of diffrax solution objects (with dense output)
    range_save : np.array (m,)
        range values to evaluate ray state at

    Returns
    -------
    full_ray_state : np.array (4, m)
        [range, travel_time, depth, pz] at each range_save point
    """
    full_ray = np.ones((3, len(range_save) - 1)) * np.nan

    for sol in sols:
        t0_seg = float(sol.ts[0])
        t1_seg = float(sol.ts[-1])
        # Use searchsorted to find indices strictly within [t0_seg, t1_seg].
        # 'left' for idx1 ensures range_save[idx1] >= t0_seg.
        # 'right' for idx2 ensures range_save[idx2-1] <= t1_seg.
        idx1 = int(np.searchsorted(range_save, t0_seg, side='left'))
        idx2 = int(np.searchsorted(range_save, t1_seg, side='right'))
        idx2 = min(idx2, len(range_save) - 1)  # leave last point for appended final state

        if idx1 >= idx2:
            continue

        values = np.array(jax.vmap(sol.evaluate)(jnp.array(range_save[idx1:idx2])))  # (N, 3)
        full_ray[:, idx1:idx2] = values.T

    final_y = np.array(sols[-1].ys[-1])
    full_ray = np.concatenate([full_ray, final_y[:, None]], axis=1)
    return np.concatenate([range_save[None, :], full_ray], axis=0)


__all__ = [
    '_shoot_ray_segment',
    '_shoot_ray_segment_fixed',
    '_shoot_single_ray_jax',
    '_shoot_rays_vmap',
    'shoot_rays',
    'shoot_ray',
    '_unpack_envi',
    '_shoot_ray_array',
]
