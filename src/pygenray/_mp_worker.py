"""
Multiprocessing worker for the CPU backend of shoot_rays().

The parent sets JAX_PLATFORM_NAME=cpu in os.environ before creating the
ProcessPoolExecutor(spawn), so spawned workers inherit it and initialize
JAX with the CPU backend (avoiding jax-metal crashes in subprocesses).
"""
import numpy as np


def _warm_up_jit(env_shapes):
    """Pool initializer: trigger JIT compilation once per worker during pool startup.

    Traces _shoot_ray_array with zero arrays of the actual environment shapes so
    workers are warm when the first real task arrives.

    Parameters
    ----------
    env_shapes : list of (shape, dtype) tuples for (cin, cpin, rin, zin, depths, dr, ba)
    """
    import numpy as np
    import jax.numpy as jnp
    from pygenray.launch_rays import _shoot_ray_array

    cin_j, cpin_j, rin_j, zin_j, d_j, dr_j, ba_j = [
        jnp.zeros(shape, dtype=dtype) for shape, dtype in env_shapes
    ]
    # Use a valid (non-zero) c value to avoid division-by-zero in the ODE
    cin_j = cin_j + 1500.0
    y0 = np.array([0.0, float(zin_j[len(zin_j)//2]), 0.0])
    _shoot_ray_array(
        y0, float(rin_j[0]), float(rin_j[-1]),
        (cin_j, cpin_j, rin_j, zin_j, d_j, dr_j), ba_j,
        1e-3, True, False,
    )


def shoot_ray_worker(args):
    """Integrate a single ray. Called in a spawned worker process.

    Reads environment arrays from shared memory (low IPC overhead even for
    GB-scale environments) and runs a single ODE integration.
    """
    from pygenray.multi_processing import _unpack_shared_memory
    from pygenray.launch_rays import _shoot_ray_array, _interpolate_ray
    import pygenray as pr
    import jax.numpy as jnp

    (array_metadata, launch_angle, source_depth, source_range,
     receiver_range, num_range_save, rtol, terminate_backwards, debug) = args

    shared_arrays, existing_shms = _unpack_shared_memory(array_metadata)
    # shared_arrays are views into the shm buffers. np.array() forces an eager copy
    # into owned numpy memory before we close the shm (jnp.array on CPU is lazy and
    # may not copy until the value is accessed, by which point the shm is unmapped).
    arrays_np = {k: np.array(v) for k, v in shared_arrays.items()}
    for shm in existing_shms.values():
        shm.close()

    cin_j           = jnp.array(arrays_np['cin'])
    cpin_j          = jnp.array(arrays_np['cpin'])
    rin_j           = jnp.array(arrays_np['rin'])
    zin_j           = jnp.array(arrays_np['zin'])
    depths_j        = jnp.array(arrays_np['depths'])
    dr_j            = jnp.array(arrays_np['depth_ranges'])
    bottom_angles_j = jnp.array(arrays_np['bottom_angle'])
    args_ode = (cin_j, cpin_j, rin_j, zin_j, depths_j, dr_j)

    # launch_angle is user-convention (positive = toward surface).
    # The serial path double-flips (once in shoot_rays, once in shoot_ray), cancelling out,
    # so y0 uses the user angle directly.
    c = pr.bilinear_interp(source_range, source_depth, rin_j, zin_j, cin_j)
    y0 = np.array([0.0, source_depth, np.sin(np.radians(launch_angle)) / float(c)])

    sols, success, n_bottom, n_surface = _shoot_ray_array(
        y0, source_range, receiver_range, args_ode, bottom_angles_j,
        rtol, terminate_backwards, debug,
    )
    if sols is None:
        return None

    range_save = np.linspace(source_range, receiver_range, num_range_save)
    full_ray = _interpolate_ray(sols, range_save)
    return pr.Ray(full_ray[0, :], full_ray[1:, :], n_bottom, n_surface, launch_angle, source_depth)
