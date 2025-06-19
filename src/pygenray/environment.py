"""
Ocean Environment Specification
Generic enironment specification for ocean sound propagation.
"""

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import scipy

class OceanEnvironment2D:
    """
    Ocean Environment Specification (2D)
    Being built for 2D ray propagation, so bottom properties aren't specified.

    Parameters
    ----------
    sound_speed : xr.DataArray
        1D or 2D array with coordinate dimensions (depth,) or (depth,range.Order of dimensions is arbitrary. Units of sound speed should be in [m/s] and units of coordinates should be in [m]. Default is range-independent Munk profile for range of 100km.
    Bathymetry : xr.DataArray
        1D array of bottom depth with coordinate dimension (range,). Units of bathymetry should be in [m]. Range grid does not have to be aligned with sound speed range grid.
        Default is flat bottom at 5000m depth.
    flat_earth_transform : bool
        whether to transform sound speed and bathymetry to flat earth.
    verbose : bool
        whether to print initialization progress messages. Default is False.

    Attributes
    ----------
    sound_speed : xr.DataArray
        2D array of sound speed with dimensions (range, depth) with units [m]
    bathymetry : xr.DataArray
        1D array of bathymetry with dimension (range,) in units [m]
    dcdz : xr.DataArray
        2D array of dc/dz with dimensions (range, depth) in units [m/s/m]
    bottom_angle : np.ndarray
        1D array of bottom slope angles in degrees at each range. The angle is computed as the arctangent of the gradient of bathymetry with respect to range. The gradient is computed usimg numpy's gradient function (2nd order)
    bottom_angle_call : callable
        function that returns the bottom slope angle in degrees at a given range. The angle is computed as the arctangent of the gradient of bathymetry with respect to range. The function takes a single argument, range in meters, and returns the angle in degrees.
    """
    
    def __init__(
            self,
            sound_speed=None,
            bathymetry=None,
            flat_earth_transform=True,
            verbose=False
            ):
        # Check Sound Speed Profile
        if sound_speed is None:
            # calculate Munk profile
            z = np.arange(0, 6000, 1)
            c_munk = munk_ssp(z)

            sound_speed = xr.DataArray(
                np.array([c_munk]*100),
                dims=['range', 'depth'],
                coords={
                    'depth': z,
                    'range': np.linspace(0, 100e3,100)
                }
            )
        else:
            # check dimension names and coordinates of sound_speed
            if not isinstance(sound_speed, xr.DataArray):
                raise TypeError("sound_speed must be an xarray DataArray.")
            if sound_speed.ndim not in [1, 2]:
                raise ValueError("sound_speed must be 1D or 2D.")
            if 'depth' not in sound_speed.dims:
                raise ValueError("sound_speed must have a 'depth' dimension.")
            if sound_speed.ndim == 2 and 'range' not in sound_speed.dims:
                raise ValueError("2D sound_speed must have a 'range' dimension.")

        # Check Bathymetry
        if bathymetry is None:
            # Default flat bottom at 5000m depth
            bathymetry = xr.DataArray(np.linspace(4500,4900,100), dims=['range'], coords={'range': np.linspace(0, 100e3,100)})

        else:
            # check dimension names and coordinates of bathymetry
            if not isinstance(bathymetry, xr.DataArray):
                raise TypeError("bathymetry must be an xarray DataArray.")
            if bathymetry.ndim != 1:
                raise ValueError("bathymetry must be 1D.")
            if 'range' not in bathymetry.dims:
                raise ValueError("bathymetry must have a 'range' dimension.")

        # If selected, do flat-earth transformation
        if flat_earth_transform:
            raise Exception('flat earth transformation not implemented yet')
            sound_speed_flat = ocean_utils.flat_earth_c(sound_speed)
            bathy_flat = ocean_utils.flat_earth_bathy(bathymetry)
            # save bathymetry and sound speed
            if verbose:
                print("Saving sound speed profile and computing dc/dz...")
            self.sound_speed = sound_speed_flat
            self.dcdz = sound_speed_flat.differentiate('depth')
            self.bathymetry = bathy_flat
        else:
            # save bathymetry and sound speed
            if verbose:
                print("Saving sound speed profile and computing dc/dz...")
            self.sound_speed = sound_speed
            self.dcdz = sound_speed.differentiate('depth').values
            self.bathymetry = bathymetry

        # compute bottom slope
        bottom_slope = np.gradient(self.bathymetry.values, self.bathymetry.range.values)
        bottom_angle_vector = np.degrees(np.arctan(bottom_slope))

        self.bottom_angle = bottom_angle_vector
        self.bottom_angle_interp= scipy.interpolate.interp1d(
            self.bathymetry.range.values,
            bottom_angle_vector,
            kind='cubic',
        )


    def plot(self,):
        '''
        plot 2D slice of environment
        '''
        fig = plt.figure(figsize=(10,3))
        ax = plt.gca()

        self.sound_speed.plot(
            x='range',
            y='depth',
            cmap='viridis',
            cbar_kwargs={'label': 'sound speed [m/s]'},
            ax=ax)
        
        # plot bathymetry
        plt.fill_between(
            self.bathymetry.range,
            self.bathymetry,
            50000,
            color='#aaaaaa',
            alpha=1,
            lw=0,
        )

        # Convert axis ticks to km
        ax.set_xlabel('range [m]')
        ax.set_ylabel('depth [m]')
        #ax.set_xticklabels([f'{x/1000:.0f}' for x in ax.get_xticks()])
        ax.set_ylim(self.sound_speed.depth.max(), self.sound_speed.depth.min())

        
        return fig

def munk_ssp(z, sofar_depth=1300, eps=0.00737):
    '''
    given vector of depth, return munk sound speed profile
    munk equations from (here)[https://web.archive.org/web/2/https://oalib-acoustics.org/website_resources/AcousticsToolbox/manual/node8.html]

    Parameters
    ----------
    z : np.array
        vector of depth
    sofar_depth : float
        depth of the SOFAR channel
    eps : float
        parameter to munk equation
    
    '''

    zh = 2*(z - sofar_depth)/sofar_depth
    c = 1500*(1 + eps*(zh - 1 + np.exp(-zh)))
    return c

def eflat(dep,lat,cs=None):
    """
    function [depf, csf]=eflat( dep, lat, cs);
    flat earth transformation
    change depths and sound speeds so that spherical shell can be
    done as an x-z slice (using WGS-84)
    """

    if cs is None:
        cs = np.zeros_like(dep)

    # WGS-84 parameters
    wgsa=6378137.0
    wgsb=6356752.314
    wgsfact=(wgsb/wgsa)**4
    Re=wgsa
    wgsa=wgsa*wgsa
    wgsb=wgsb*wgsb

    ll=np.pi*lat/180.0

    ree1=wgsa/np.sqrt(wgsa*np.cos(ll)*np.cos(ll)+wgsb*np.sin(ll)*np.sin(ll))
    re=ree1*np.sqrt(np.cos(ll)*np.cos(ll)+wgsfact*np.sin(ll)*np.sin(ll))

    E=dep/re
    depf=dep*(1.0 + E*(0.50 + E/3.0))
    csf=cs*(1.0+E*(1.0+E))

    return depf, csf

def eflatinv(depf, lat, csf=None):
    """
    Inverse flat earth transformation
    
    Parameters:
        depf : np.array
            depth in flat earth coordinates
        lat : np.array
            latitude
        csf : np.array
            optional speed factor
    Returns:
        dep : np.array
            transformed depth
        cs : np.array
            transformed sound speed
    """
    
    # Ensure inputs are column vectors
    depf = np.reshape(depf, (-1,))
    lat = np.reshape(lat, (-1,))
    
    # Default parameter handling
    if csf is None:
        csf = np.zeros(depf.shape)
    csf = np.reshape(csf, (-1,))
    
    # WGS-84 parameters
    wgsa = 6378137.0
    wgsb = 6356752.314
    wgsfact = (wgsb/wgsa)**4
    Re = wgsa
    wgsa = wgsa*wgsa
    wgsb = wgsb*wgsb
    
    # Calculate Earth radius at given latitude
    ll = np.pi*lat/180.0
    ree1 = wgsa/np.sqrt(wgsa*np.cos(ll)*np.cos(ll) + wgsb*np.sin(ll)*np.sin(ll))
    re = ree1*np.sqrt(np.cos(ll)*np.cos(ll) + wgsfact*np.sin(ll)*np.sin(ll))
    
    # Define accuracy for ridder function
    zacc = 0.001*np.ones(depf.shape)
    
    # Provide a better bracket for the root finder
    # We know true depth is less than flat depth, so try a range that will work
    bracket_lower = depf * 0.5  # Try half the flat depth
    bracket_upper = depf        # Upper bound is the flat depth
    
    # Call ridder function to solve for depth
    try:
        dep = _ridder(eflat, bracket_lower, bracket_upper, depf, zacc, lat)[0]
    except ValueError:
        # If that fails, try a wider range
        bracket_lower = depf * 0.1
        try:
            dep = _ridder(eflat, bracket_lower, bracket_upper, depf, zacc, lat)[0]
        except ValueError:
            # If all else fails, use an approximation
            # This is a rough approximation, but better than failing
            dep = depf / (1.0 + 0.5*(depf/re) + (depf/re)**2/3.0)
    
    # Final calculations
    E = dep/re
    cs = csf/(1.0 + E*(1.0 + E))
    
    return dep, cs

def _ridder(fhdl, xl, xh, xrhs, xacc, *args):
    """
    Solves f(x)=xrhs using Ridder's method
    
    Parameters:
        fhdl: function handle
        xl, xh: lower and upper bounds
        xrhs: right-hand side value
        xacc: desired accuracy
        *args: additional arguments to pass to fhdl
    """
    
    # Initial function evaluations
    fl = fhdl(xl, *args) - xrhs
    fh = fhdl(xh, *args) - xrhs
    
    # Check that root is bracketed
    if np.any(fl * fh > 0):
        raise ValueError('root must be bracketed')
    
    # Initial midpoint
    x = (xl + xh) / 2
    fx = fhdl(x, *args) - xrhs
    
    # Main iteration loop
    while True:
        xm = (xl + xh) / 2
        fm = fhdl(xm, *args) - xrhs
        dnm = np.sqrt(fm * fm - fl * fh)
        if np.any(dnm == 0): 
            return x, fx
        xnew = xm + (xm - xl) * np.sign(fl - fh) * fm / dnm
        
        if np.all(abs(xnew - x) <= xacc): 
            return x, fx
        
        x = xnew
        fnew = fhdl(x, *args) - xrhs
        fx = fnew
        if np.all(fnew == 0): 
            return x, fx
        
        ind = np.where(fnew * fm < 0)
        xl = np.copy(xl)  # Create copies to avoid in-place modifications of arrays
        fl = np.copy(fl)
        xh = np.copy(xh)
        fh = np.copy(fh)
        
        xl[ind] = xm[ind]
        fl[ind] = fm[ind]
        xh[ind] = xnew[ind]
        fh[ind] = fnew[ind]
        
        ind = np.where(fnew * fh < 0)
        xl[ind] = xnew[ind]
        fl[ind] = fnew[ind]
        
        ind = np.where(fnew * fl < 0)
        xh[ind] = xnew[ind]
        fh[ind] = fnew[ind]
        
        if np.all(abs(xh - xl) <= xacc): 
            return x, fx
        
# Public API
__all__ = ['OceanEnvironment2D']