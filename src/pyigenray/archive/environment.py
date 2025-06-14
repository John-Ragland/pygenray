"""
Ocean Environment Specification
Generic enironment specification for ocean sound propagation.
This module is intended to be a placeholder, to be eventually replaced by a 
uniform environmental interface that's compatible with all ocean sound propagation models.

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

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
            # TODO
            pass
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
            self.bathymetry = xr.DataArray(np.full((2,), 5000), dims=['range'], coords={'range': [0, 100e3]})
        else:
            # check dimension names and coordinates of bathymetry
            if not isinstance(bathymetry, xr.DataArray):
                raise TypeError("bathymetry must be an xarray DataArray.")
            if bathymetry.ndim != 1:
                raise ValueError("bathymetry must be 1D.")
            if 'range' not in bathymetry.dims:
                raise ValueError("bathymetry must have a 'range' dimension.")

        # Do flat-earth transformation
        if flat_earth_transform:
            self.flat_earth_transform()

        # save bathymetry and sound speed
        if verbose:
            print("Saving sound speed profile and computing dc/dz...")
        self.sound_speed = sound_speed
        self.c = sound_speed.values
        self.cp = sound_speed.differentiate('depth').values
        self.cr = sound_speed.range.values
        self.cz = sound_speed.depth.values
        self.bathymetry = bathymetry


    def flat_earth_transform(self, ):
        '''
        flat_earth_transform
        Transform sound speed and bathymetry to flat earth coordinates.
        '''
        return
    
    def plot(self,):
        '''
        plot 2D slice of environment
        '''
        fig = plt.figure(figsize=(10,6))
        ax = plt.gca()

        self.sound_speed.plot(
            x='range',
            y='depth',
            cmap='viridis',
            cbar_kwargs={'label': 'sound speed [m/s]'},
            ax=ax)
        
        # Convert axis ticks to km
        ax.set_xlabel('range [km]')
        ax.set_ylabel('depth [m]')
        ax.set_xticklabels([f'{x/1000:.0f}' for x in ax.get_xticks()])
        
        return fig