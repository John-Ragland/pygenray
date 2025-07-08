from multiprocessing import shared_memory
import numpy as np
import uuid
import os

def _init_shared_memory(
        cin,
        cpin,
        rin,
        zin,
        depths,
        depth_ranges,
        bottom_angle
):
    '''
    Initialize shared memory for multiprocessing

    Parameters
    ----------
    cpin : np.array (m,n)
        2D array of dc/dz
    rin : np.array (m,)
        range coordinate for c arrays
    zin : np.array (n,)
        depth coordinate for c arrays
    depths : np.array(m,)
        array of depths (meters), should be 1D with shape (k,)
    depth_ranges : np.array(k,)
        array of depth ranges (meters), should be 1D with shape (k,)
    bottom_angle : np.array (k,)
        array of bottom angles (degrees), should be 1D with shape (k,) and values should align with depth_ranges coordinate.

    Returns
    -------
    array_metadata : dict
        Dictionary containing metadata of shared memory arrays
    shms : dict
        Dictionary containing shared memory objects for each array
    '''

    # Create unique id, uses PID and shotened UUID to handle SLURM better
    unique_id = f'{os.getpid()}_{uuid.uuid4().hex[:8]}'

    shared_array_name_bases = [
        'cin','cpin','rin','zin','depths','depth_ranges','bottom_angle'
    ]

    shared_array_names = [f'{name}_{unique_id}' for name in shared_array_name_bases]

    shared_arrays_np = {
        f'cin_{unique_id}':cin,
        f'cpin_{unique_id}':cpin,
        f'rin_{unique_id}':rin,
        f'zin_{unique_id}':zin,
        f'depths_{unique_id}':depths,
        f'depth_ranges_{unique_id}':depth_ranges,
        f'bottom_angle_{unique_id}':bottom_angle,
    }

    shms = {}
    shared_arrays = {}
    array_metadata = {}

    for var in shared_arrays_np:
        shms[var] = shared_memory.SharedMemory(create=True, size=shared_arrays_np[var].nbytes, name=var)
        shared_arrays[var] = np.ndarray(shared_arrays_np[var].shape, dtype=shared_arrays_np[var].dtype, buffer=shms[var].buf)
        shared_arrays[var][:] = shared_arrays_np[var][:]
        array_metadata[var] = {
            'shape': shared_arrays_np[var].shape,
            'dtype': shared_arrays_np[var].dtype,
        }
    
    return array_metadata, shms

def _unpack_shared_memory(shared_array_metadata):
    """
    Unpack shared memory arrays from metadata

    Parameters
    ----------
    shared_array_metadata : dict
        Dictionary containing metadata of shared memory arrays

    Returns
    -------
    shared_arrays : dict
        Dictionary containing unpacked shared memory arrays with base names as keys
    existing_shms : dict
        Dictionary containing existing shared memory objects with unique names as keys
    """

    existing_shms = {}
    shared_arrays = {}
    
    # Define the mapping from base names to what we expect
    base_names = ['cin', 'cpin', 'rin', 'zin', 'depths', 'depth_ranges', 'bottom_angle']
    
    for var in shared_array_metadata:
        # Access shared memory with the unique name
        existing_shms[var] = shared_memory.SharedMemory(name=var)
        
        # Create numpy array
        array = np.ndarray(
            shared_array_metadata[var]['shape'],
            dtype=shared_array_metadata[var]['dtype'], 
            buffer=existing_shms[var].buf)
        
        # Map back to base name for easy access
        for base_name in base_names:
            if var.startswith(f'{base_name}_'):
                shared_arrays[base_name] = array
                break
        
    return shared_arrays, existing_shms

__all__ = ['_init_shared_memory', '_unpack_shared_memory']