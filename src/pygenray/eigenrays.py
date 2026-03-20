"""
Tools and methods for calculating eigenrays for specifed receiver depths.
"""
import functools
import warnings

import pygenray as pr
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from tqdm import tqdm

from .launch_rays import _shoot_single_ray_jax


def find_eigenrays(
        rays,
        receiver_depths,
        source_depth,
        source_range,
        receiver_range,
        num_range_save,
        environment,
        ztol=1,
        max_iter=20,
        num_workers=None,
        parallel=True,
        max_bounces=50,
        **kwargs,
    ):
    '''
    Given an initial ray fan, find eigenrays with [regula falsi](https://en.wikipedia.org/wiki/Regula_falsi#The_regula_falsi_(false_position)_method) method of root finding.

    Parameters
    ----------
    rays : pr.RayFan
        RayFan object containing sweep of rays to be used for finding eigenrays.
    receiver_depths : array like
        one dimensional array, or list containing receiver depths
    source_depth : float
        source depth in meters
    source_range : float
        source range in meters
    receiver_range : float
        receiver range in meters
    num_range_save : int
        number of range values to save the ray state at
    environment : pr.OceanEnvironment2D
        OceanEnvironment2D object containing environment parameters for ray tracing.
    ztol : float, optional
        depth tolerance for eigenrays, by default 1 m
    max_iter : int, optional
        maximum number of root finding iterations, default 20
    num_workers : int, optional
        deprecated; ignored (vmap is used instead of multiprocessing)
    parallel : bool, optional
        if True (default), use jax.vmap for parallel eigenray finding;
        if False, use serial loop
    max_bounces : int, optional
        maximum number of bounces per ray (vmap path only), default 50
    kwargs : keyword arguments
        additional keyword arguments passed to pr.shoot_ray (serial path only)

    Returns
    -------
    erays : pr.EigenRays
    '''
    if num_workers is not None:
        warnings.warn(
            'num_workers is deprecated and ignored; eigenray finding now uses jax.vmap.',
            DeprecationWarning, stacklevel=2,
        )

    from .launch_rays import _unpack_envi

    erays_dict = {}
    num_eigenrays = {}
    num_eigenrays_found = {}
    failed_eray_theta_brackets = {}

    # Unpack environment once for the vmap path
    if parallel:
        flatearth = kwargs.get('flatearth', True)
        cin, cpin, rin, zin, depths, depth_ranges, bottom_angles = _unpack_envi(environment, flatearth=flatearth)
        cin_j = jnp.array(cin)
        cpin_j = jnp.array(cpin)
        rin_j = jnp.array(rin)
        zin_j = jnp.array(zin)
        depths_j = jnp.array(depths)
        dr_j = jnp.array(depth_ranges)
        bottom_angles_j = jnp.array(bottom_angles)
        env_arrays = (cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j, bottom_angles_j)
        range_save = np.linspace(source_range, receiver_range, num_range_save)
        range_save_j = jnp.array(range_save, dtype=jnp.float64)

    for rd_idx, receiver_depth in enumerate(receiver_depths):
        ## get initial bracketing rays
        depth_sign = np.sign(rays.zs[:, -1] + receiver_depth)
        sign_change = np.diff(depth_sign)
        bracket_idxs_start = np.where(sign_change)[0]

        num_eigenrays[receiver_depth] = len(bracket_idxs_start)

        bracket_idxs = np.column_stack([bracket_idxs_start, bracket_idxs_start + 1])

        z1s = rays.zs[bracket_idxs[:, 0].astype(int), -1]
        z2s = rays.zs[bracket_idxs[:, 1].astype(int), -1]
        theta1s = rays.thetas[bracket_idxs[:, 0].astype(int)]
        theta2s = rays.thetas[bracket_idxs[:, 1].astype(int)]

        erays_dict[rd_idx] = []
        failed_eray_theta_brackets[rd_idx] = []

        if len(z1s) == 0:
            num_eigenrays_found[rd_idx] = 0
            continue

        if parallel:
            outputs_k, n_bottoms_k, n_surfaces_k, thetas_k, convergeds_k = _find_eigenrays_vmap(
                z1s, z2s, theta1s, theta2s,
                float(receiver_depth), float(source_depth), float(source_range), float(receiver_range),
                range_save_j, env_arrays,
                max_iter=max_iter, ztol=float(ztol), max_bounces=max_bounces,
                rtol=kwargs.get('rtol', 1e-9),
                terminate_backwards=kwargs.get('terminate_backwards', True),
            )
            outputs_np = np.array(outputs_k)
            n_bottoms_np = np.array(n_bottoms_k)
            n_surfaces_np = np.array(n_surfaces_k)
            thetas_np = np.array(thetas_k)
            convergeds_np = np.array(convergeds_k)

            for k in range(len(z1s)):
                if convergeds_np[k]:
                    y = outputs_np[k].T  # (3, N) ODE format
                    ray = pr.Ray(
                        r=range_save,
                        y=y,
                        n_bottom=int(n_bottoms_np[k]),
                        n_surface=int(n_surfaces_np[k]),
                        launch_angle=float(thetas_np[k]),
                        source_depth=float(source_depth),
                    )
                    erays_dict[rd_idx].append(ray)
                else:
                    failed_eray_theta_brackets[rd_idx].append((float(theta1s[k]), float(theta2s[k])))

        else:
            # Serial path
            regula_falsi_thetas = theta1s - (z1s + receiver_depth) * (theta2s - theta1s) / (z2s - z1s)
            for k in tqdm(range(len(regula_falsi_thetas)), desc='Finding eigenrays:'):
                ray = _find_single_eigenray((
                    k, z1s[k], z2s[k], theta1s[k], theta2s[k], regula_falsi_thetas[k],
                    receiver_depth, source_depth, source_range, receiver_range,
                    num_range_save, environment, ztol, max_iter, kwargs
                ))
                if ray is not None:
                    erays_dict[rd_idx].append(ray)
                else:
                    failed_eray_theta_brackets[rd_idx].append((theta1s[k], theta2s[k]))

        num_eigenrays_found[rd_idx] = len(erays_dict[rd_idx])

    erays = pr.EigenRays(
        receiver_depths, erays_dict, environment,
        num_eigenrays, num_eigenrays_found, failed_eray_theta_brackets
    )
    return erays


def _find_eigenrays_vmap(
    z1s, z2s, theta1s, theta2s,  # (K,) numpy arrays, user convention
    receiver_depth, source_depth, source_range, receiver_range,
    range_save_j,   # (N,) JAX float64
    env_arrays,
    max_iter=20, ztol=1.0, max_bounces=50, rtol=1e-9, terminate_backwards=True,
):
    """
    vmap eigenray finding over K brackets for a single receiver depth.

    Returns
    -------
    outputs : (K, N, 3) float64 — ODE format rays
    n_bottoms : (K,) int32
    n_surfaces : (K,) int32
    thetas_converged : (K,) float64 — launch angles (user convention degrees)
    convergeds : (K,) bool
    """
    z1s_j = jnp.array(z1s, dtype=jnp.float64)
    z2s_j = jnp.array(z2s, dtype=jnp.float64)
    theta1s_j = jnp.array(theta1s, dtype=jnp.float64)
    theta2s_j = jnp.array(theta2s, dtype=jnp.float64)

    _single = functools.partial(
        _find_single_eigenray_jax,
        receiver_depth=jnp.float64(receiver_depth),
        source_depth=jnp.float64(source_depth),
        source_range=jnp.float64(source_range),
        receiver_range=jnp.float64(receiver_range),
        range_save=range_save_j,
        env_arrays=env_arrays,
        max_iter=max_iter,
        ztol=ztol,
        max_bounces=max_bounces,
        rtol=rtol,
        terminate_backwards=terminate_backwards,
    )
    vmapped = jax.vmap(_single, in_axes=(0, 0, 0, 0))
    outputs, n_bottoms, n_surfaces, thetas_converged, convergeds = vmapped(
        z1s_j, z2s_j, theta1s_j, theta2s_j
    )
    return outputs, n_bottoms, n_surfaces, thetas_converged, convergeds


@functools.partial(jax.jit, static_argnames=['max_iter', 'max_bounces', 'rtol', 'terminate_backwards'])
def _find_single_eigenray_jax(
    z1, z2, theta1, theta2,  # scalar float64, user convention (z negative downward, theta degrees)
    receiver_depth,           # scalar float64, positive downward
    source_depth, source_range, receiver_range,
    range_save,               # (N,) float64
    env_arrays,
    max_iter=20,              # static
    ztol=1.0,
    max_bounces=50,           # static
    rtol=1e-9,                # static
    terminate_backwards=True, # static
):
    """
    Find a single eigenray using regula falsi in a lax.while_loop.

    Parameters
    ----------
    z1, z2 : scalar float64
        Bracketing ray depths in user convention (ray.z, negative for downward).
    theta1, theta2 : scalar float64
        Bracketing launch angles in user convention (degrees).
    receiver_depth : scalar float64
        Target depth, positive downward (e.g., 500.0 for 500 m depth).
    source_depth, source_range, receiver_range : scalar float64
    range_save : (N,) float64
    env_arrays : tuple of JAX arrays
    max_iter, max_bounces, rtol, terminate_backwards : static args

    Returns
    -------
    last_output : (N, 3) float64 — ODE format ray path of converged eigenray
    n_bottom : int32
    n_surface : int32
    theta_converged : float64 — launch angle (user convention degrees) of converged ray
    converged : bool
    """
    N = range_save.shape[0]

    # Initial regula falsi guess
    theta_rf0 = theta1 - (z1 + receiver_depth) * (theta2 - theta1) / (z2 - z1)

    init_carry = (
        z1,
        z2,
        theta1,
        theta2,
        jnp.int32(0),                          # iter_count
        theta_rf0,                              # theta_rf (current guess)
        jnp.full((N, 3), jnp.nan, dtype=jnp.float64),  # last_output
        jnp.int32(0),                          # last_n_bottom
        jnp.int32(0),                          # last_n_surface
        jnp.float64(theta_rf0),               # theta_converged
        jnp.bool_(False),                      # converged
    )

    def _cond_fn(carry):
        *_, converged = carry
        iter_count = carry[4]
        return ~converged & (iter_count < max_iter)

    def _body_fn(carry):
        (z1, z2, theta1, theta2, iter_count, theta_rf,
         last_output, last_n_bottom, last_n_surface, theta_conv, converged) = carry

        # Shoot ray at current regula falsi angle.
        # The serial path calls shoot_ray(theta_rf) which internally flips:
        #   launch_angle = -theta_rf → y0[2] = sin(-theta_rf_rad)/c
        # _shoot_single_ray_jax does NOT flip, so we negate here to match.
        launch_angle_rad = jnp.radians(-theta_rf)
        output, n_bottom, n_surface, alive, reached = _shoot_single_ray_jax(
            launch_angle_rad, source_depth, source_range, receiver_range,
            range_save, env_arrays,
            max_bounces=max_bounces, rtol=rtol, terminate_backwards=terminate_backwards,
        )

        ray_ok = alive & reached

        # Depth at receiver in user convention (ray.z = -z_ode)
        z_end_user = -output[-1, 1]

        # Check convergence
        converged_new = ray_ok & (jnp.abs(z_end_user + receiver_depth) < ztol)

        # Update bracket only when ray succeeded
        on_z1_side = jnp.sign(z_end_user + receiver_depth) == jnp.sign(z1 + receiver_depth)
        z1_new = jnp.where(ray_ok & on_z1_side, z_end_user, z1)
        theta1_new = jnp.where(ray_ok & on_z1_side, theta_rf, theta1)
        z2_new = jnp.where(ray_ok & ~on_z1_side, z_end_user, z2)
        theta2_new = jnp.where(ray_ok & ~on_z1_side, theta_rf, theta2)

        dz = z2_new - z1_new
        # Guard against degenerate bracket (avoid div-by-zero)
        safe_dz = jnp.where(jnp.abs(dz) < 1e-10, jnp.float64(1e-10), dz)
        theta_rf_new = theta1_new - (z1_new + receiver_depth) * (theta2_new - theta1_new) / safe_dz

        # Record converged state
        last_output_new = jnp.where(converged_new, output, last_output)
        last_n_bottom_new = jnp.where(converged_new, n_bottom, last_n_bottom)
        last_n_surface_new = jnp.where(converged_new, n_surface, last_n_surface)
        theta_conv_new = jnp.where(converged_new, theta_rf, theta_conv)

        return (
            z1_new, z2_new, theta1_new, theta2_new,
            iter_count + jnp.int32(1),
            theta_rf_new,
            last_output_new,
            last_n_bottom_new,
            last_n_surface_new,
            theta_conv_new,
            converged_new,
        )

    final_carry = lax.while_loop(_cond_fn, _body_fn, init_carry)
    (_, _, _, _, _, _, last_output, last_n_bottom, last_n_surface,
     theta_conv, converged) = final_carry

    return last_output, last_n_bottom, last_n_surface, theta_conv, converged


def _find_single_eigenray(args):
    """
    Find single Eigen ray given the bracketing ray depths, and launch angles.
    """
    k, z1, z2, theta1, theta2, regula_falsi_theta, receiver_depth, source_depth, source_range, receiver_range, num_range_save, environment, ztol, max_iter, kwargs = args

    iter_count = 0
    # Regula Falsi root finding loop
    while True:

        ray = pr.shoot_ray(source_depth, source_range, regula_falsi_theta, receiver_range, num_range_save, environment, **kwargs)

        if ray is None:
            print(f'Failed to find eigen ray for receiver depth {receiver_depth} [m] and approximate launch angle {regula_falsi_theta} [m] ray θ = 90°')
            return None

        if np.abs(ray.z[-1] + receiver_depth) < ztol:

            # flip launch angle to match sign convention
            ray.launch_angle = -ray.launch_angle
            return ray

        # Ray is on z1 side of receiver
        if np.sign(ray.z[-1] + receiver_depth) == np.sign(z1 + receiver_depth):
            z1 = ray.z[-1]
            theta1 = regula_falsi_theta
        # Ray is on z2 side of receiver
        else:
            z2 = ray.z[-1]
            theta2 = regula_falsi_theta

        regula_falsi_theta = theta1 - (z1 + receiver_depth) * (theta2 - theta1) / (z2 - z1)

        if iter_count > max_iter:
            return None

        iter_count += 1


__all__ = ['find_eigenrays']
