from multiprocessing import shared_memory
import numpy as np

def _init_shared_memory(
        cin,
        cpin,
        rin,
        zin,
        depths,
        depth_ranges,
        bottom_angle,
        x_eval
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
    shared_array_names = [
        'cin','cpin','rin','zin','depths','depth_ranges','bottom_angle','x_eval'
    ]
    shared_arrays_np = {
        'cin':cin,
        'cpin':cpin,
        'rin':rin,
        'zin':zin,
        'depths':depths,
        'depth_ranges':depth_ranges,
        'bottom_angle':bottom_angle,
        'x_eval':x_eval
    }

    shms = {}
    shared_arrays = {}
    array_metadata = {}
    # clean up shared arrays
    _cleanup_shared_memory(shared_array_names)

    for var in shared_arrays_np:
        shms[var] = shared_memory.SharedMemory(create=True, size=shared_arrays_np[var].nbytes, name=var)
        shared_arrays[var] = np.ndarray(shared_arrays_np[var].shape, dtype=shared_arrays_np[var].dtype, buffer=shms[var].buf)
        shared_arrays[var][:] = shared_arrays_np[var][:]
        array_metadata[var] = {
            'shape': shared_arrays_np[var].shape,
            'dtype': shared_arrays_np[var].dtype,
        }
    
    return array_metadata, shms

def _cleanup_shared_memory(names):
    """
    Clean up existing shared memory objects by names
    
    Parameters
    ----------
    names : list
        names of shared memory objects to clean up
    """
    for name in names:
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
            #print(f"Cleaned up existing shared memory: {name}")
        except FileNotFoundError:
            # Memory doesn't exist, which is fine
            pass
        except Exception as e:
            print(f"Error cleaning up {name}: {e}")

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
        Dictionary containing unpacked shared memory arrays
    existing_shms : dict
        Dictionary containing existing shared memory objects
    """
    shared_array_names = [
        'cin','cpin','rin','zin','depths','depth_ranges','bottom_angle','x_eval'
    ]

    existing_shms = {}
    shared_arrays = {}
    for var in shared_array_names:
        existing_shms[var] = shared_memory.SharedMemory(name=var)

        shared_arrays[var] = np.ndarray(
            shared_array_metadata[var]['shape'],
            dtype=shared_array_metadata[var]['dtype'], buffer=existing_shms[var].buf)
        
    return shared_arrays, existing_shms

__all__ = ['_init_shared_memory', '_cleanup_shared_memory', '_unpack_shared_memory']