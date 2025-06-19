import pygenray as pr
import numpy as np
import xarray as xr

def rayintrd_bathy(
        y0s : np.array,
        dx : float,
        r : float,
        c : np.array,
        cp : np.array,
        cpp : np.array,
        z : np.array,
        xin : np.array,
        D : float,
        xbot : np.array,
        bot : np.array,
        bot_slope : np.array
):
    """"
    Integrate ray for range dependant bathymetry
    This function is intended to be called from matlab and is built to match the input and output of the matlab ray tracing code `rayintrd_bathy`.
    
    Different source depths are not supported, so y[1,:] (*or y(2,:) in matlab indexing*) must all be the same value. y[1,0] (y(2,1) in matlab indexing) is used for all ray initial conditions.

    .. note::
        If you intend to use python, you should use either {func}`pygenray.shoot_ray` or {func}`pygenray.shoot_rays`.

    Parameters
    ----------
    y0s : np.array
        array of shape (3,j) containing all ray initial conditions.
    dx : float
        range distance to save ray state at
    r : float
        integration range in meters
    c : np.array
        array of shape (m,n) specifying sound speed.
    cp : np.array
        array of shape (m,n) specifying sound speed derivate dc/dz. **not used** cp is numerically calculated from c
    cpp : np.array
        array of shape (m,n) specifying double sound speed derivate d^2c/dz^2. **not used**: ray calculation doesn't use cpp
    z : np.array
        array of shape (n,) specifying depth coordinates for sound speed arrays
    xin : np.array
        array of shape (n,) specifying dpeth coordinates for sound speed arrays
    D : float
        water depth. **not currently used**
    xbot : np.array
        array of shape (k,) specifying range-dependant bathymetry ranges
    bot : np.array
        array of shape (k,) specifying range-dependant bathymetry depths
    bot_slope : np.array
        array of shape (k,) specifying range_dependant bathymetry slopes. **this array is not used in this function**. The bottom slope is calculated from xbot and bot.

    Returns
    -------
    xray : np.array
        array of the ray x values
    tray : np.array
        array of the ray time values
    zray : np.array
        array of ray depth values
    pray : np.array
        array of ray parameter variables (sinθ/c)
    dzdpz : np.array
        **not calculated** None is returned
    """
    # Convert all inputs to numpy arrays with explicit copy and contiguous memory
    y0s = np.ascontiguousarray(np.asarray(y0s, dtype=np.float64))
    c = np.ascontiguousarray(np.asarray(c, dtype=np.float64))
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    xin = np.ascontiguousarray(np.asarray(xin, dtype=np.float64))
    xbot = np.ascontiguousarray(np.asarray(xbot, dtype=np.float64))
    bot = np.ascontiguousarray(np.asarray(bot, dtype=np.float64))
    bot_slope = np.ascontiguousarray(np.asarray(bot_slope, dtype=np.float64))

    # Ensure scalar values are proper Python types
    dx = float(dx)
    r = float(r)
    D = float(D)
    
    launch_angles = []
    # calculate ray angles
    for k in range(y0s.shape[1]):
        theta, c0 = pr.ray_angle(0, y0s[:,k], c, xin, z)
        launch_angles.append(theta)
    
    # Construct environment
    soundspeed = xr.DataArray(c, dims=['range','depth'], coords={'range':xin, 'depth':z})
    bathymetry = xr.DataArray(bot, dims=['range'], coords={'range':xbot})
    envi = pr.OceanEnvironment2D(soundspeed, bathymetry, flat_earth_transform=False)

    # calculate x_eval
    x_eval = np.arange(0, r, dx)

    rays = pr.shoot_rays(y0s[1,0], 0, launch_angles, r, x_eval, envi)
    xray = rays.x
    tray = rays.t
    zray = rays.z
    pray = rays.p
    dzpdz = None

    return xray, tray, zray, pray, dzpdz

__all__ = ['rayintrd_bathy']