import time
import pygenray as pr

def toy_process(idx : int, array_metadata : dict):
    shared_arrays, existing_shms = pr._unpack_shared_memory(shared_array_metadata=array_metadata)
    time.sleep(1)
    value = shared_arrays['cin'][idx,0]
    # unlink all shared arrays after process is done
    for var in existing_shms:
        existing_shms[var].close()
    return value

__all__ = ['toy_process']