"""
Tools for generic JAX use for geospatial data.
This module may be wrapped into a seperate package later.
"""

from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=['order','mode'])
def interp_physical_coords(data, coords, query_points, order=1, mode='nearest'):
    """
    Interpolate data at physical coordinates for arbitrary dimensions using map_coordinates
    
    Parameters:
    -----------
    data: jnp.ndarray of shape (M₁, M₂, ..., Mₖ)
        K-dimensional data array to interpolate from
    coords: sequence of K arrays
        Each coords[i] is an array of shape (Mᵢ,) giving physical coordinates along axis i
    query_points: sequence of K arrays
        Each query_points[i] contains coordinates to query along axis i
        All arrays must be broadcastable to the same shape
    order: int, default=1
        Interpolation order (1=linear, 0=nearest, etc.)
    mode: str, default='nearest'
        How to handle boundaries ('nearest', 'wrap', 'reflect', etc.)
        
    Returns:
    --------
    jnp.ndarray: Interpolated values at the query points
    """
    # Convert physical coordinates to array indices
    indices = []
    for coord_array, query in zip(coords, query_points):
        # Map from physical coordinates to array indices (0...len-1)
        idx = jnp.interp(query, coord_array, jnp.arange(len(coord_array)))
        indices.append(idx)
    
    # Stack indices into required format for map_coordinates: (ndim, *output_shape)
    indices_array = jnp.stack(indices)
    
    # Perform the interpolation
    return jax.scipy.ndimage.map_coordinates(data, indices_array, order=order, mode=mode)