"""
experimenting with implementing things in jax
"""
import jax
import jax.numpy as jnp
import pyigenray as pr

@jax.jit
def bilinear_interp_jx(r, z, cin, rin, zin):
    '''
    Optimized bilinear interpolation of 2D array cin at point (r,z)
    '''
    # 1. Use more efficient index finding
    r_idx = jnp.clip(jnp.searchsorted(rin, r) - 1, 0, rin.shape[0] - 2)
    z_idx = jnp.clip(jnp.searchsorted(zin, z) - 1, 0, zin.shape[0] - 2)
    
    # 2. Pre-fetch indices for better memory access patterns
    r_idx_next = r_idx + 1
    z_idx_next = z_idx + 1
    
    # 3. Get bounding points (use direct indexing)
    r0 = rin[r_idx]
    r1 = rin[r_idx_next]
    z0 = zin[z_idx]
    z1 = zin[z_idx_next]
    
    # 4. Calculate weights with bounds protection
    dr = r1 - r0
    dz = z1 - z0
    # Using where ensures no division by zero
    wr = jnp.where(dr > 0, (r - r0) / dr, 0.0)
    wz = jnp.where(dz > 0, (z - z0) / dz, 0.0)
    
    # 5. Get corner values (one lookup each)
    c00 = cin[r_idx, z_idx]
    c01 = cin[r_idx, z_idx_next]
    c10 = cin[r_idx_next, z_idx]
    c11 = cin[r_idx_next, z_idx_next]
    
    # 6. Optimized bilinear formula (compute once, reuse values)
    omwr = 1.0 - wr
    omwz = 1.0 - wz
    
    # Compute in a way that minimizes temporary variables
    result = omwr * omwz * c00 + \
             omwr * wz * c01 + \
             wr * omwz * c10 + \
             wr * wz * c11

    return result

@jax.jit
def derivsrd_jx(
        x : float,
        y : jnp.array,
        args : tuple
    ) -> jnp.array:
    '''
    differential equations for ray propagation.

    Parameters
    ----------
    x : float
        horizontal range (kilometers)
    y : jnp.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    args : tuple (6,)
        cin : jnp.array (m,n)
            2D array of sound speed values
        cpin : jnp.array (m,n)
            2D array of dc/dz
        rin : jnp.array (m,)
            range coordinate for c arrays
        zin : jnp.array (n,)
            depth coordinate for c arrays
        depths : jnp.array (k,)
            array of depths (meters), should be 1D with shape (k,)
        depth_ranges : jnp.array (k,)
            array of depth ranges (kilometers), should be 1D with shape (k,)

    Returns
    -------
    dydx : np.array (3,)
        derivative of ray variables with respect to horizontal range, [dT/dx, dz/dx, dp/dx]
    '''
    # unpack args
    cin, cpin, rin, zin,_,_ = args
    
    #unpack ray variables
    z=y[1] # current depth
    pz=y[2] # current ray parameter

    #interpolate sound speed and its derivative at current depth and range
    c = pr.interp_physical_coords(cin, (rin, zin), (x,z))
    cp = pr.interp_physical_coords(cpin, (rin, zin), (x,z))
    
    #c = bilinear_interp(x, z, cin, rin, zin)
    #cp = bilinear_interp(x, z, cpin, rin, zin)

    # calculate derivatives
    fact=1/jnp.sqrt(1-(c**2)*(pz**2))
    dydx = jnp.array([
        fact/c,
        c*pz*fact,
        -fact*cp/(c**2)
    ])

    return dydx

@jax.jit
def bottom_bounce_jx(x,y,args):
    """
    Parameters
    ----------
    x : float
        horizontal range (kilometers)
    y : jnp.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]
    args : tuple (6,) (only last two are used)
        cin : jnp.array (m,n)
            2D array of sound speed values
        cpin : jnp.array (m,n)
            2D array of dc/dz
        rin : jnp.array (m,)
            range coordinate for c arrays
        zin : jnp.array (n,)
            depth coordinate for c arrays
        depth : jnp.array(m,)
            array of depths (meters), should be 1D with shape (m,)
        depth_ranges : jnp.array(m,)
            array of depth ranges (kilometers), should be 1D with shape (m,)

    Returns
    -------
    value : np.array (2,)
        bounce event values for surface and bottom
    isterminal : np.array (2,)
        boolean array indicating if the event should stop the integration
    direction : np.array (2,)
        direction of the event, 1 for increasing, -1 for decreasing
    """

    # unpack args
    _,_,_,_,depths,depth_ranges  = args

    water_depth = pr.interp_physical_coords(depths, [depth_ranges], [x])
    ray_depth = y[1]

    # in original code, bottom bounce is enforced within 2 meters
    # I'm removing this for now, which might cause numerical issues
    #if jnp.abs(val2) < 2:
    #    val2=0

    # crosses zero for bottom reflection
    bottom_term = ray_depth-water_depth

    return bottom_term  # bounce event at surface and bottom

@jax.jit
def surface_bounce_jx(x,y):
    """
    Parameters
    ----------
    x : float (unused)
        horizontal range (kilometers)
    y : jnp.array (3,)
        ray variables, [travel time, depth, ray parameter (sin(θ)/c)]

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
